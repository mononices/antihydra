# Local correction for pwn/saint_vespers_and_the_copper_choir.
#
# The shipped build/Dockerfile installs the compiler with --no-install-recommends.
# On Ubuntu libc6-dev only arrives as a *Recommends* of gcc, so the C headers are
# missing and the compile step dies with:
#
#     /build/vespers.c:10:10: fatal error: stdio.h: No such file or directory
#
# Only the build stage's package list changed (libc6-dev added); everything else
# is byte-identical to the dataset file. overrides.toml points at this copy and
# the build runs from a staged context, so competition/ stays read-only.
FROM ubuntu:22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        make \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY src/vespers.c /build/vespers.c

RUN gcc -O2 -fstack-protector-all -D_FORTIFY_SOURCE=2 \
        -Wl,-z,relro,-z,now -pie -fPIE \
        -o /build/vespers /build/vespers.c \
    && strip /build/vespers

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        socat \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -d /home/vespers -s /usr/sbin/nologin vespers

WORKDIR /home/vespers
COPY --from=build /build/vespers /home/vespers/vespers
COPY build/flag.txt /home/vespers/flag.txt
RUN chmod 555 /home/vespers/vespers && \
    chown root:root /home/vespers/flag.txt && \
    chmod 444 /home/vespers/flag.txt

EXPOSE 5000
USER vespers
CMD ["socat", "-T30", "TCP-LISTEN:5000,reuseaddr,fork", "EXEC:/home/vespers/vespers,stderr"]
