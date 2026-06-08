"""
Generador de tráfico de red sintético basado en comportamiento real de botnets CTU-13.

Genera 6 tipos de flujos con distintos grados de dificultad de detección:
  1. Botnet UDP/DNS C&C        - Neris-like,  fácil         (~2 pkts, 300 bytes, CON)
  2. Botnet TCP escaneo puertos - Rbot-like,   fácil-medio   (S_/S_R, 1 pkt, 60 bytes)
  3. Botnet IRC keepalive       - Rbot-like,   medio         (FSPA_FSPA, largo)
  4. Botnet HTTP C&C mimético   - Virut-like,  difícil       (tamaños similares a HTTP normal)
  5. Normal DNS                 - parece botnet DNS pero es legítimo
  6. Normal navegación web      - TCP con FSPA_FSPA, grande, lento
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2024)


def r(lo, hi, n, log=False):
    """Uniform entre lo y hi. Si log=True opera en espacio log."""
    if log:
        return np.exp(RNG.uniform(np.log(lo), np.log(hi), n))
    return RNG.uniform(lo, hi, n)


def ri(lo, hi, n):
    return RNG.integers(lo, hi + 1, n)


def choice(options, weights, n):
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return RNG.choice(options, n, p=weights)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Botnet UDP/DNS C&C — Neris (escenario 9)
#    Flujos UDP muy cortos hacia servidor DNS que es en realidad el C&C.
#    src hace pregunta (~72 bytes), servidor responde (~230 bytes).
#    pps alto (2 pkts / 0.15 s ≈ 13 pps), src_ratio bajo (~0.24).
#    DIFICULTAD BAJA: el patrón es casi idéntico al real.
# ──────────────────────────────────────────────────────────────────────────────
def gen_botnet_dns(n=2500):
    dur       = r(0.005, 0.8, n)
    tot_pkts  = ri(2, 4, n)
    src_bytes = ri(60, 90, n).astype(float)
    # respuesta DNS es más grande; bytes totales ~ src + respuesta
    extra     = ri(160, 320, n).astype(float)
    tot_bytes = src_bytes + extra

    return pd.DataFrame({
        "dur":       dur,
        "proto":     "udp",
        "dir":       choice(["  <->", "   ->"], [0.85, 0.15], n),
        "state":     "CON",
        "stos":      0.0,
        "dtos":      0.0,
        "tot_pkts":  tot_pkts.astype(float),
        "tot_bytes": tot_bytes,
        "src_bytes": src_bytes,
        "label":     "flow=From-Botnet-Synthetic-UDP-DNS",
    })


# ──────────────────────────────────────────────────────────────────────────────
# 2. Botnet TCP escaneo de puertos — Rbot
#    Intentos SYN-only hacia IPs/puertos aleatorios.
#    dur ≈ 0, 1 pkt, 60 bytes.  Estado S_ (SYN enviado, sin respuesta) o S_R
#    (SYN + RST recibido).  pps muy alto, bytes_per_pkt muy bajo.
#    DIFICULTAD MEDIA: S_ es una señal fuerte pero el modelo lo sabe.
# ──────────────────────────────────────────────────────────────────────────────
def gen_botnet_portscan(n=1500):
    dur       = r(0.0001, 2.0, n, log=True)
    tot_pkts  = ri(1, 3, n).astype(float)
    src_bytes = ri(40, 80, n).astype(float)
    tot_bytes = src_bytes + RNG.integers(0, 50, n)

    states = choice(["S_", "S_R", "S_RA", "RA_"], [0.50, 0.25, 0.15, 0.10], n)

    return pd.DataFrame({
        "dur":       dur,
        "proto":     "tcp",
        "dir":       "   ->",
        "state":     states,
        "stos":      0.0,
        "dtos":      0.0,
        "tot_pkts":  tot_pkts,
        "tot_bytes": tot_bytes.astype(float),
        "src_bytes": src_bytes,
        "label":     "flow=From-Botnet-Synthetic-TCP-Scan",
    })


# ──────────────────────────────────────────────────────────────────────────────
# 3. Botnet IRC keepalive — Rbot (sesiones largas al C&C IRC)
#    Sesiones TCP largas (minutos a horas) con tráfico periódico pequeño.
#    El truco: parecen conexiones HTTPS largas pero src_ratio es ~0.5
#    y bytes_per_pkt es muy bajo (solo headers IRC, ~80 bytes).
#    DIFICULTAD MEDIA-ALTA: la duración larga es inusual para botnet típico.
# ──────────────────────────────────────────────────────────────────────────────
def gen_botnet_irc(n=1000):
    dur       = r(60, 3600, n)           # sesiones largas
    tot_pkts  = r(20, 800, n, log=True)
    # Paquetes IRC muy pequeños ~80 bytes promedio → bytes_per_pkt bajo
    bytes_per_pkt = r(60, 150, n)
    tot_bytes = tot_pkts * bytes_per_pkt
    # Tráfico casi simétrico (bot envía comandos, C&C responde parecido)
    src_ratio = r(0.35, 0.65, n)
    src_bytes = tot_bytes * src_ratio

    return pd.DataFrame({
        "dur":       dur,
        "proto":     "tcp",
        "dir":       "  <->",
        "state":     choice(["FSPA_FSPA", "FSPA_FSA", "FSA_FSPA"], [0.70, 0.20, 0.10], n),
        "stos":      0.0,
        "dtos":      0.0,
        "tot_pkts":  tot_pkts,
        "tot_bytes": tot_bytes,
        "src_bytes": src_bytes,
        "label":     "flow=From-Botnet-Synthetic-IRC",
    })


# ──────────────────────────────────────────────────────────────────────────────
# 4. Botnet HTTP C&C mimético — Virut-like (más difícil de detectar)
#    El bot usa puerto 80/443 y hace peticiones HTTP que parecen browsing.
#    Diferencias sutiles vs. tráfico normal:
#      - Periodismo muy regular (bots tienen timer fijo, humanos no)
#      - src_ratio más bajo que navegación real (respuestas del C&C son pequeñas)
#      - bytes_per_pkt en el rango bajo-medio
#    DIFICULTAD ALTA: muy parecido a tráfico web normal.
# ──────────────────────────────────────────────────────────────────────────────
def gen_botnet_http_cc(n=1200):
    dur       = r(0.1, 15.0, n)
    tot_pkts  = r(4, 40, n, log=True)
    # Respuesta HTTP del C&C es modesta: ~500-3000 bytes
    bytes_per_pkt = r(100, 600, n)
    tot_bytes = tot_pkts * bytes_per_pkt
    # Bot envía request pequeño, recibe respuesta mediana
    src_ratio = r(0.05, 0.30, n)
    src_bytes = tot_bytes * src_ratio

    return pd.DataFrame({
        "dur":       dur,
        "proto":     "tcp",
        "dir":       "  <->",
        "state":     choice(["FSPA_FSPA", "FSPA_FSA", "SPA_FSPA", "FSA_FSA"],
                            [0.55, 0.20, 0.15, 0.10], n),
        "stos":      0.0,
        "dtos":      0.0,
        "tot_pkts":  tot_pkts,
        "tot_bytes": tot_bytes,
        "src_bytes": src_bytes,
        "label":     "flow=From-Botnet-Synthetic-HTTP-CC",
    })


# ──────────────────────────────────────────────────────────────────────────────
# 5. Normal DNS — el más parecido a botnet (la trampa)
#    Consultas DNS legítimas: mismos proto/state/dir que botnet DNS,
#    pero más variación en bytes y timing menos periódico.
#    PROPÓSITO: ver si el modelo genera falsos positivos con DNS normal.
# ──────────────────────────────────────────────────────────────────────────────
def gen_normal_dns(n=2000):
    # Loopback / caché local: respuesta en microsegundos, imposible en C&C remoto
    dur       = r(0.00005, 0.00009, n)
    tot_pkts  = ri(2, 6, n).astype(float)
    src_bytes = ri(40, 120, n).astype(float)
    extra     = ri(80, 500, n).astype(float)    # respuestas DNS varían más
    tot_bytes = src_bytes + extra

    return pd.DataFrame({
        "dur":       dur,
        "proto":     "udp",
        "dir":       choice(["  <->", "   ->"], [0.75, 0.25], n),
        "state":     choice(["CON", "INT"], [0.90, 0.10], n),
        "stos":      0.0,
        "dtos":      0.0,
        "tot_pkts":  tot_pkts,
        "tot_bytes": tot_bytes,
        "src_bytes": src_bytes,
        "label":     "flow=Normal-Synthetic-DNS",
    })


# ──────────────────────────────────────────────────────────────────────────────
# 6. Normal navegación web — el más alejado de botnet
#    TCP con handshake completo, transferencia de datos grande, duración variable.
#    bytes_per_pkt alto (>1000), src_ratio bajo (descargando más de lo que sube).
# ──────────────────────────────────────────────────────────────────────────────
def gen_normal_web(n=2500):
    dur       = r(0.05, 120, n, log=True)
    tot_pkts  = r(3, 5000, n, log=True)
    bytes_per_pkt = r(200, 1500, n)
    tot_bytes = tot_pkts * bytes_per_pkt
    # Usuario descarga más de lo que sube (streaming, páginas, etc.)
    src_ratio = r(0.01, 0.25, n)
    src_bytes = tot_bytes * src_ratio

    return pd.DataFrame({
        "dur":       dur,
        "proto":     "tcp",
        "dir":       "  <->",
        "state":     choice(["FSPA_FSPA", "FSA_FSPA", "FSPA_FSA", "FSA_FSA", "INT"],
                            [0.50, 0.20, 0.15, 0.10, 0.05], n),
        "stos":      0.0,
        "dtos":      0.0,
        "tot_pkts":  tot_pkts,
        "tot_bytes": tot_bytes,
        "src_bytes": src_bytes,
        "label":     "flow=Normal-Synthetic-Web",
    })


# ──────────────────────────────────────────────────────────────────────────────
# Combinar todo y guardar
# ──────────────────────────────────────────────────────────────────────────────
def main():
    frames = [
        gen_botnet_dns(2500),
        gen_botnet_portscan(1500),
        gen_botnet_irc(1000),
        gen_botnet_http_cc(1200),
        gen_normal_dns(2000),
        gen_normal_web(2500),
    ]

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n_bot  = df["label"].str.contains("Botnet").sum()
    n_norm = df["label"].str.contains("Normal").sum()

    print(f"Total flujos: {len(df)}")
    print(f"  Botnet : {n_bot}  ({n_bot/len(df)*100:.1f}%)")
    print(f"  Normal : {n_norm}  ({n_norm/len(df)*100:.1f}%)")
    print()
    print("Distribución por tipo:")
    for lbl, cnt in df["label"].value_counts().items():
        print(f"  {lbl:50s}  {cnt:5d}")

    out = "synthetic_test.parquet"
    df.to_parquet(out, index=False)
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
