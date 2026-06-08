"""
Generador de escenarios de tráfico CUALITATIVAMENTE distintos.

Cada escenario modela una situación de red diferente:
  mirai        — IoT botnet (UDP flood + escaneo telnet), fácil de detectar
  apt          — APT sigiloso (HTTPS beacon + DNS tunelizado), muy difícil
  spam         — Botnet spam (SMTP masivo + IRC C&C), detección parcial
  corporativo  — Red empresarial con exfiltración de datos, botnet minoría
  p2p          — Cryptominer + botnet IoT en red doméstica

Los datasets difieren en:
  - proporción botnet / normal (15 % a 55 %)
  - protocolo dominante (UDP-heavy vs TCP-heavy)
  - familias de botnet simuladas
  - tipo de tráfico normal que acompaña
  - dificultad de detección esperada por el modelo
"""

import numpy as np
import pandas as pd
import sys

RNG = np.random.default_rng(0)   # solo para reproducibilidad del generador base


def rng(seed):
    return np.random.default_rng(seed)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def ru(lo, hi, n, r, log=False):
    if log:
        return np.exp(r.uniform(np.log(max(lo, 1e-9)), np.log(hi), n))
    return r.uniform(lo, hi, n)


def ri(lo, hi, n, r):
    return r.integers(lo, hi + 1, n)


def ch(opts, w, n, r):
    w = np.array(w, float)
    w /= w.sum()
    return r.choice(opts, n, p=w)


def frame(n, proto, dir_, state, dur, pkts, tot_bytes, src_bytes, label):
    return pd.DataFrame({
        "dur":       dur,
        "proto":     proto if isinstance(proto, np.ndarray) else np.full(n, proto),
        "dir":       dir_  if isinstance(dir_,  np.ndarray) else np.full(n, dir_),
        "state":     state if isinstance(state, np.ndarray) else np.full(n, state),
        "stos":      0.0,
        "dtos":      0.0,
        "tot_pkts":  pkts.astype(float),
        "tot_bytes": tot_bytes.astype(float),
        "src_bytes": src_bytes.astype(float),
        "label":     label,
    })


# ═════════════════════════════════════════════════════════════════════════════
# ESCENARIO 1 — MIRAI  (IoT botnet)
# Botnet: ~50 %  —  fácil de detectar
#
# Mirai infecta routers, cámaras y dispositivos IoT.
# Comportamiento:  (a) UDP flood hacia IPs aleatorias — dir unidireccional,
#                  pps altísimo, bytes pequeños.
#                  (b) Telnet scan (TCP port 23) — SYN-only, 1 paquete.
#                  (c) HTTP scan — SYN o handshake mínimo.
# Normal: dispositivos IoT (heartbeats UDP pequeños), DNS, web básica.
# Diferencia clave vs modelo entrenado: UDP flood tiene dir="->" y pps>>100,
# muy distinto al UDP/CON bidireccional de Neris DNS.
# ═════════════════════════════════════════════════════════════════════════════
def escenario_mirai(n_bot=3000, n_norm=3000, seed=1):
    r = rng(seed)

    # ── Botnet: UDP flood (alto pps, unidireccional)
    n1 = n_bot // 2
    pkts1 = ri(10, 500, n1, r).astype(float)
    dur1  = ru(0.001, 0.5, n1, r)
    bpp1  = ru(64, 128, n1, r)                    # paquetes pequeños de flood
    tb1   = pkts1 * bpp1
    bot_flood = frame(n1, "udp", "   ->", "INT",
                      dur1, pkts1, tb1, tb1, "flow=From-Botnet-Mirai-UDP-Flood")

    # ── Botnet: telnet / HTTP scan (TCP SYN-only)
    n2   = n_bot - n1
    dur2 = ru(0.0001, 0.05, n2, r, log=True)
    pkts2 = np.ones(n2)
    tb2   = ri(40, 74, n2, r).astype(float)
    state2 = ch(["S_","S_R","RA_"], [0.6, 0.3, 0.1], n2, r)
    bot_scan = frame(n2, "tcp", "   ->", state2,
                     dur2, pkts2, tb2, tb2, "flow=From-Botnet-Mirai-TCP-Scan")

    # ── Normal: heartbeat IoT (UDP, muy regular, pequeño)
    n3   = n_norm // 3
    dur3 = ru(0.0001, 0.002, n3, r)
    pkts3 = np.full(n3, 2.0)
    tb3   = ri(64, 96, n3, r).astype(float)
    sb3   = (tb3 * 0.5).astype(float)
    norm_iot = frame(n3, "udp", "  <->", "CON",
                     dur3, pkts3, tb3, sb3, "flow=Normal-IoT-Heartbeat")

    # ── Normal: DNS rápido
    n4 = n_norm // 3
    dur4  = ru(0.00005, 0.0009, n4, r)
    pkts4 = np.full(n4, 2.0)
    tb4   = ri(120, 280, n4, r).astype(float)
    sb4   = ri(40, 80, n4, r).astype(float)
    norm_dns = frame(n4, "udp", "  <->", "CON",
                     dur4, pkts4, tb4, sb4, "flow=Normal-DNS")

    # ── Normal: web básica
    n5   = n_norm - n3 - n4
    dur5 = ru(0.1, 30, n5, r, log=True)
    pkts5 = ru(4, 200, n5, r, log=True)
    bpp5  = ru(300, 1200, n5, r)
    tb5   = pkts5 * bpp5
    sr5   = ru(0.02, 0.2, n5, r)
    norm_web = frame(n5, "tcp", "  <->",
                     ch(["FSPA_FSPA","FSA_FSPA","FSPA_FSA"],[0.6,0.2,0.2], n5, r),
                     dur5, pkts5, tb5, tb5 * sr5, "flow=Normal-Web")

    df = pd.concat([bot_flood, bot_scan, norm_iot, norm_dns, norm_web],
                   ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# ESCENARIO 2 — APT SIGILOSO  (Advanced Persistent Threat)
# Botnet: ~22 %  —  muy difícil de detectar
#
# Un APT exfiltra datos lentamente sin levantar alarmas.
# Comportamiento:  (a) Beacon HTTPS: TCP FSPA_FSPA, tamaños y duración
#                  idénticos a navegación normal, src_ratio muy bajo.
#                  (b) DNS tunneling: UDP/CON con payloads MÁS GRANDES
#                  que DNS normal (datos codificados en el nombre DNS).
# Normal: red empresarial heavy HTTPS, mucho DNS, poco UDP.
# Diferencia clave: el botnet intenta ser indistinguible del tráfico web.
# El modelo probablemente no lo detecta bien.
# ═════════════════════════════════════════════════════════════════════════════
def escenario_apt(n_bot=1500, n_norm=5500, seed=2):
    r = rng(seed)

    # ── Botnet: HTTPS beacon (parece navegación web normal)
    n1   = n_bot * 2 // 3
    dur1 = ru(0.2, 8.0, n1, r)
    pkts1 = ru(5, 35, n1, r, log=True)
    bpp1  = ru(400, 1400, n1, r)           # tamaño de paquete realista HTTPS
    tb1   = pkts1 * bpp1
    sr1   = ru(0.02, 0.08, n1, r)         # cliente envía poco (request pequeño)
    bot_beacon = frame(n1, "tcp", "  <->",
                       ch(["FSPA_FSPA","FSA_FSPA"],[0.7,0.3], n1, r),
                       dur1, pkts1, tb1, tb1 * sr1,
                       "flow=From-Botnet-APT-HTTPS-Beacon")

    # ── Botnet: DNS tunneling (payloads más grandes que DNS normal)
    n2   = n_bot - n1
    dur2 = ru(0.05, 3.0, n2, r)
    pkts2 = ri(2, 8, n2, r).astype(float)
    # Los datos van codificados en las preguntas DNS → paquetes más grandes
    sb2   = ri(150, 500, n2, r).astype(float)
    tb2   = sb2 + ri(200, 600, n2, r).astype(float)
    bot_dns_tunnel = frame(n2, "udp", "  <->", "CON",
                           dur2, pkts2, tb2, sb2,
                           "flow=From-Botnet-APT-DNS-Tunnel")

    # ── Normal: HTTPS empresarial (heavy — downloads grandes)
    n3   = n_norm * 2 // 5
    dur3 = ru(0.1, 60, n3, r, log=True)
    pkts3 = ru(5, 3000, n3, r, log=True)
    bpp3  = ru(500, 1500, n3, r)
    tb3   = pkts3 * bpp3
    sr3   = ru(0.01, 0.12, n3, r)         # descargando mucho más de lo que sube
    norm_https = frame(n3, "tcp", "  <->",
                       ch(["FSPA_FSPA","FSA_FSPA","FSPA_FSA"],[0.55,0.25,0.2], n3, r),
                       dur3, pkts3, tb3, tb3 * sr3, "flow=Normal-Enterprise-HTTPS")

    # ── Normal: DNS empresarial (rápido, pequeño)
    n4   = n_norm // 4
    dur4 = ru(0.00005, 0.001, n4, r)
    pkts4 = ri(2, 4, n4, r).astype(float)
    sb4   = ri(30, 80, n4, r).astype(float)
    tb4   = sb4 + ri(60, 200, n4, r).astype(float)
    norm_dns = frame(n4, "udp", "  <->", "CON",
                     dur4, pkts4, tb4, sb4, "flow=Normal-Enterprise-DNS")

    # ── Normal: sesiones internas TCP (file share, DB, etc.)
    n5   = n_norm - n3 - n4
    dur5 = ru(0.05, 120, n5, r, log=True)
    pkts5 = ru(3, 1000, n5, r, log=True)
    bpp5  = ru(200, 1200, n5, r)
    tb5   = pkts5 * bpp5
    sr5   = ru(0.1, 0.6, n5, r)           # más simétrico que descarga web pura
    norm_internal = frame(n5, "tcp", "  <->",
                          ch(["FSPA_FSPA","FSA_FSA","INT"],[0.5,0.3,0.2], n5, r),
                          dur5, pkts5, tb5, tb5 * sr5, "flow=Normal-Internal-TCP")

    df = pd.concat([bot_beacon, bot_dns_tunnel, norm_https, norm_dns, norm_internal],
                   ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# ESCENARIO 3 — SPAM BOTNET  (similar a Neris pero enfocado en SMTP)
# Botnet: ~42 %  —  detección parcial
#
# Botnets de spam (como Rustock o Cutwail) abren miles de conexiones TCP
# hacia servidores de correo (puerto 25). La mayoría son rechazadas (S_)
# porque las IPs están en listas negras. Algunas logran entregar spam.
# También mantienen IRC para control.
# Normal: correo legítimo, web, DNS. Más tráfico TCP que en otros escenarios.
# Diferencia clave: muchos flujos TCP S_ (parecidos al escaneo de Rbot) pero
# mezclados con SMTP real y mucho más variación en bytes.
# ═════════════════════════════════════════════════════════════════════════════
def escenario_spam(n_bot=2500, n_norm=3500, seed=3):
    r = rng(seed)

    # ── Botnet: intentos SMTP rechazados (IP en blacklist)
    n1    = n_bot // 2
    dur1  = ru(0.01, 5.0, n1, r)
    pkts1 = ri(1, 4, n1, r).astype(float)
    tb1   = ri(40, 200, n1, r).astype(float)
    state1 = ch(["S_","S_R","S_RA"],[0.55,0.30,0.15], n1, r)
    bot_smtp_reject = frame(n1, "tcp", "   ->", state1,
                            dur1, pkts1, tb1, tb1,
                            "flow=From-Botnet-Spam-SMTP-Rejected")

    # ── Botnet: spam entregado (conexión completa con entrega de mensaje)
    n2   = n_bot // 4
    dur2 = ru(5, 120, n2, r)
    pkts2 = ri(8, 40, n2, r).astype(float)
    # Email de spam: src envía el mensaje, por eso src_bytes alto
    sr2   = ru(0.55, 0.85, n2, r)         # bot envía más de lo que recibe
    bpp2  = ru(200, 800, n2, r)
    tb2   = pkts2 * bpp2
    bot_smtp_sent = frame(n2, "tcp", "  <->",
                          ch(["FSPA_FSPA","FSPA_FSA"],[0.7,0.3], n2, r),
                          dur2, pkts2, tb2, tb2 * sr2,
                          "flow=From-Botnet-Spam-SMTP-Sent")

    # ── Botnet: IRC C&C (keepalive periódico)
    n3   = n_bot - n1 - n2
    dur3 = ru(60, 3600, n3, r)
    pkts3 = ru(15, 600, n3, r, log=True)
    bpp3  = ru(60, 150, n3, r)
    tb3   = pkts3 * bpp3
    sr3   = ru(0.35, 0.65, n3, r)
    bot_irc = frame(n3, "tcp", "  <->",
                    ch(["FSPA_FSPA","FSA_FSPA"],[0.75,0.25], n3, r),
                    dur3, pkts3, tb3, tb3 * sr3,
                    "flow=From-Botnet-Spam-IRC-CC")

    # ── Normal: email legítimo (tamaños variados, flujos completos)
    n4   = n_norm // 3
    dur4 = ru(1, 60, n4, r)
    pkts4 = ri(6, 50, n4, r).astype(float)
    bpp4  = ru(150, 600, n4, r)
    tb4   = pkts4 * bpp4
    sr4   = ru(0.05, 0.4, n4, r)
    norm_mail = frame(n4, "tcp", "  <->",
                      ch(["FSPA_FSPA","FSA_FSA"],[0.7,0.3], n4, r),
                      dur4, pkts4, tb4, tb4 * sr4, "flow=Normal-Legitimate-Email")

    # ── Normal: DNS
    n5   = n_norm // 4
    dur5 = ru(0.00005, 0.0009, n5, r)
    pkts5 = np.full(n5, 2.0)
    tb5   = ri(120, 300, n5, r).astype(float)
    sb5   = ri(40, 90, n5, r).astype(float)
    norm_dns = frame(n5, "udp", "  <->", "CON",
                     dur5, pkts5, tb5, sb5, "flow=Normal-DNS")

    # ── Normal: web
    n6   = n_norm - n4 - n5
    dur6 = ru(0.05, 60, n6, r, log=True)
    pkts6 = ru(3, 500, n6, r, log=True)
    bpp6  = ru(300, 1400, n6, r)
    tb6   = pkts6 * bpp6
    sr6   = ru(0.02, 0.2, n6, r)
    norm_web = frame(n6, "tcp", "  <->",
                     ch(["FSPA_FSPA","FSA_FSPA","FSPA_FSA"],[0.55,0.25,0.2], n6, r),
                     dur6, pkts6, tb6, tb6 * sr6, "flow=Normal-Web")

    df = pd.concat([bot_smtp_reject, bot_smtp_sent, bot_irc,
                    norm_mail, norm_dns, norm_web], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# ESCENARIO 4 — RED CORPORATIVA CON EXFILTRACIÓN
# Botnet: ~14 %  —  difícil (botnet es minoría en tráfico denso)
#
# Una APT ya instalada exfiltra datos lentamente.
# Comportamiento:  (a) Exfiltración: TCP largo con src_ratio ALTO (el bot
#                  envía más de lo que recibe — inusual en tráfico normal).
#                  (b) HTTP C&C: peticiones pequeñas y frecuentes.
# Normal: red empresarial densa: HTTPS pesado, VoIP UDP, file sync, DNS.
# El botnet es solo el 14 % del tráfico — desafía el --balance del modelo.
# Diferencia clave: src_ratio alto en el botnet es la señal, pero el modelo
# no fue entrenado para detectar eso específicamente.
# ═════════════════════════════════════════════════════════════════════════════
def escenario_corporativo(n_bot=1000, n_norm=6000, seed=4):
    r = rng(seed)

    # ── Botnet: exfiltración (el bot sube datos → src_ratio alto)
    n1   = n_bot * 2 // 3
    dur1 = ru(30, 600, n1, r)
    pkts1 = ru(20, 2000, n1, r, log=True)
    bpp1  = ru(400, 1200, n1, r)
    tb1   = pkts1 * bpp1
    # Bot envía los datos robados → src_bytes >> dst_bytes
    sr1   = ru(0.60, 0.92, n1, r)
    bot_exfil = frame(n1, "tcp", "  <->",
                      ch(["FSPA_FSPA","FSPA_FSA"],[0.7,0.3], n1, r),
                      dur1, pkts1, tb1, tb1 * sr1,
                      "flow=From-Botnet-Corp-Exfiltration")

    # ── Botnet: HTTP C&C (peticiones pequeñas regulares a servidor de control)
    n2   = n_bot - n1
    dur2 = ru(0.1, 3.0, n2, r)
    pkts2 = ru(4, 20, n2, r, log=True)
    bpp2  = ru(100, 400, n2, r)
    tb2   = pkts2 * bpp2
    sr2   = ru(0.03, 0.15, n2, r)
    bot_cc = frame(n2, "tcp", "  <->",
                   ch(["FSPA_FSPA","FSPA_FSA"],[0.65,0.35], n2, r),
                   dur2, pkts2, tb2, tb2 * sr2,
                   "flow=From-Botnet-Corp-HTTP-CC")

    # ── Normal: HTTPS empresarial masivo
    n3   = n_norm * 2 // 5
    dur3 = ru(0.1, 90, n3, r, log=True)
    pkts3 = ru(5, 5000, n3, r, log=True)
    bpp3  = ru(500, 1500, n3, r)
    tb3   = pkts3 * bpp3
    sr3   = ru(0.01, 0.1, n3, r)          # descargando: src_ratio bajo
    norm_https = frame(n3, "tcp", "  <->",
                       ch(["FSPA_FSPA","FSA_FSPA","FSPA_FSA"],[0.5,0.3,0.2], n3, r),
                       dur3, pkts3, tb3, tb3 * sr3, "flow=Normal-Corp-HTTPS")

    # ── Normal: VoIP (UDP, pps alto, bytes pequeños, duración larga)
    n4   = n_norm // 6
    dur4 = ru(10, 3600, n4, r)
    pkts4 = ru(500, 50000, n4, r, log=True)
    bpp4  = ru(40, 200, n4, r)            # paquetes RTP pequeños
    tb4   = pkts4 * bpp4
    sr4   = ru(0.4, 0.6, n4, r)          # VoIP es simétrico
    norm_voip = frame(n4, "udp", "  <->", "CON",
                      dur4, pkts4, tb4, tb4 * sr4, "flow=Normal-Corp-VoIP")

    # ── Normal: file sync (grandes transferencias bidireccionales)
    n5   = n_norm // 6
    dur5 = ru(5, 300, n5, r)
    pkts5 = ru(100, 10000, n5, r, log=True)
    bpp5  = ru(800, 1500, n5, r)
    tb5   = pkts5 * bpp5
    sr5   = ru(0.3, 0.7, n5, r)          # sincronización: relativamente simétrico
    norm_sync = frame(n5, "tcp", "  <->",
                      ch(["FSPA_FSPA","FSA_FSA"],[0.7,0.3], n5, r),
                      dur5, pkts5, tb5, tb5 * sr5, "flow=Normal-Corp-FileSync")

    # ── Normal: DNS
    n6   = n_norm - n3 - n4 - n5
    dur6 = ru(0.00005, 0.001, n6, r)
    pkts6 = ri(2, 4, n6, r).astype(float)
    tb6   = ri(100, 300, n6, r).astype(float)
    sb6   = ri(30, 80, n6, r).astype(float)
    norm_dns = frame(n6, "udp", "  <->", "CON",
                     dur6, pkts6, tb6, sb6, "flow=Normal-Corp-DNS")

    df = pd.concat([bot_exfil, bot_cc,
                    norm_https, norm_voip, norm_sync, norm_dns], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# ESCENARIO 5 — RED DOMÉSTICA: CRYPTOMINER + BOTNET P2P
# Botnet: ~35 %  —  detección media
#
# Un cryptominer abre conexiones TCP largas y simétricas al pool de minería.
# Un botnet P2P (tipo Gameover Zeus) hace conexiones UDP y TCP cortas a
# muchos pares distintos.
# Normal: streaming de video (TCP enorme, src muy bajo), juegos online (UDP
# bidireccional, pps medio), DNS doméstico.
# Diferencia clave: el cryptominer (src_ratio ~0.5, largo) y el botnet P2P
# (muchas conexiones cortas) son distintos de todo lo que vio el modelo.
# ═════════════════════════════════════════════════════════════════════════════
def escenario_p2p(n_bot=2000, n_norm=3700, seed=5):
    r = rng(seed)

    # ── Botnet: cryptominer (pool mining TCP, largo, simétrico, alto bandwidth)
    n1   = n_bot // 2
    dur1 = ru(300, 7200, n1, r)            # sesiones de horas
    pkts1 = ru(1000, 50000, n1, r, log=True)
    bpp1  = ru(100, 400, n1, r)            # bloques de trabajo pequeños
    tb1   = pkts1 * bpp1
    sr1   = ru(0.4, 0.6, n1, r)           # muy simétrico (envía nonces, recibe shares)
    bot_miner = frame(n1, "tcp", "  <->",
                      ch(["FSPA_FSPA","FSA_FSA"],[0.75,0.25], n1, r),
                      dur1, pkts1, tb1, tb1 * sr1,
                      "flow=From-Botnet-CryptoMiner")

    # ── Botnet: P2P (UDP y TCP cortos a muchos peers)
    n2a  = (n_bot - n1) * 2 // 3    # UDP P2P
    dur2a = ru(0.01, 2.0, n2a, r)
    pkts2a = ri(1, 6, n2a, r).astype(float)
    tb2a   = ri(100, 500, n2a, r).astype(float)
    sr2a   = ru(0.3, 0.7, n2a, r)
    bot_p2p_udp = frame(n2a, "udp", "  <->", "CON",
                        dur2a, pkts2a, tb2a, tb2a * sr2a,
                        "flow=From-Botnet-P2P-UDP")

    n2b  = n_bot - n1 - n2a    # TCP P2P cortos
    dur2b = ru(0.1, 10, n2b, r)
    pkts2b = ri(4, 20, n2b, r).astype(float)
    bpp2b  = ru(100, 500, n2b, r)
    tb2b   = pkts2b * bpp2b
    sr2b   = ru(0.3, 0.7, n2b, r)
    bot_p2p_tcp = frame(n2b, "tcp", "  <->",
                        ch(["FSPA_FSPA","FSA_FSPA"],[0.65,0.35], n2b, r),
                        dur2b, pkts2b, tb2b, tb2b * sr2b,
                        "flow=From-Botnet-P2P-TCP")

    # ── Normal: streaming de video (TCP, descarga masiva, src_ratio muy bajo)
    n3   = n_norm // 3
    dur3 = ru(5, 3600, n3, r, log=True)
    pkts3 = ru(100, 100000, n3, r, log=True)
    bpp3  = ru(1000, 1500, n3, r)          # paquetes grandes (video)
    tb3   = pkts3 * bpp3
    sr3   = ru(0.001, 0.02, n3, r)         # casi nada sube (solo ACKs)
    norm_stream = frame(n3, "tcp", "  <->",
                        ch(["FSPA_FSPA","FSA_FSPA"],[0.7,0.3], n3, r),
                        dur3, pkts3, tb3, tb3 * sr3,
                        "flow=Normal-VideoStreaming")

    # ── Normal: juegos online (UDP bidireccional, pps medio, pequeño)
    n4   = n_norm // 4
    dur4 = ru(60, 3600, n4, r)
    pkts4 = ru(500, 20000, n4, r, log=True)
    bpp4  = ru(50, 300, n4, r)             # paquetes de juego pequeños
    tb4   = pkts4 * bpp4
    sr4   = ru(0.35, 0.65, n4, r)         # relativamente simétrico
    norm_game = frame(n4, "udp", "  <->", "CON",
                      dur4, pkts4, tb4, tb4 * sr4,
                      "flow=Normal-OnlineGaming")

    # ── Normal: DNS doméstico
    n5   = n_norm - n3 - n4
    dur5 = ru(0.00005, 0.001, n5, r)
    pkts5 = ri(2, 4, n5, r).astype(float)
    tb5   = ri(100, 280, n5, r).astype(float)
    sb5   = ri(30, 80, n5, r).astype(float)
    norm_dns = frame(n5, "udp", "  <->", "CON",
                     dur5, pkts5, tb5, sb5, "flow=Normal-Home-DNS")

    df = pd.concat([bot_miner, bot_p2p_udp, bot_p2p_tcp,
                    norm_stream, norm_game, norm_dns], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
ESCENARIOS = {
    "escenario_mirai":        (escenario_mirai,       "IoT Mirai flood + telnet scan"),
    "escenario_apt":          (escenario_apt,         "APT HTTPS beacon + DNS tunnel"),
    "escenario_spam":         (escenario_spam,        "Spam botnet SMTP + IRC C&C"),
    "escenario_corporativo":  (escenario_corporativo, "Exfiltración en red empresarial"),
    "escenario_p2p":          (escenario_p2p,         "Cryptominer + botnet P2P doméstico"),
}


def main():
    for nombre, (fn, desc) in ESCENARIOS.items():
        df  = fn()
        out = f"{nombre}.parquet"
        df.to_parquet(out, index=False)

        n_bot  = df["label"].str.contains("Botnet").sum()
        n_norm = len(df) - n_bot
        pct    = n_bot / len(df) * 100

        print(f"\n{out}  —  {desc}")
        print(f"  Total: {len(df):,}  |  Botnet: {n_bot:,} ({pct:.0f}%)  "
              f"|  Normal: {n_norm:,} ({100-pct:.0f}%)")
        for lbl, cnt in df["label"].value_counts().items():
            tag = "BOT " if "Botnet" in lbl else "NORM"
            print(f"  {tag}  {lbl.split('=')[-1]:<40s}  {cnt:>5,}")


if __name__ == "__main__":
    main()
