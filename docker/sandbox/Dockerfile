# antihydra sandbox image: general-purpose CTF toolbox for LLM agents.
#
# Build:
#   docker build -t ctfenv:antihydra docker/sandbox
#
# Heavy optional layers can be disabled:
#   docker build -t ctfenv:antihydra \
#     --build-arg WITH_HEAVY=0 --build-arg WITH_TORCH=0 docker/sandbox

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND} \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# system packages
RUN dpkg --add-architecture i386 \
 && apt-get update \
 && apt-get install -y \
      bash coreutils curl wget ca-certificates netcat-openbsd socat nmap \
      git jq unzip zip file vim less bsdmainutils \
      build-essential cmake make pkg-config \
      gcc-multilib g++-multilib libc6-dev-i386 \
      gdb gdbserver ltrace strace binutils \
      python3 python3-dev python3-pip python3-venv \
      libssl-dev libffi-dev zlib1g-dev libjpeg-dev libpng-dev libtiff-dev libbz2-dev \
      libgmp-dev libmpfr-dev libeigen3-dev libfplll-dev libz3-dev libhwloc-dev \
      binwalk sleuthkit foremost testdisk dcfldd xfsprogs pngcheck \
      libimage-exiftool-perl steghide imagemagick \
      tesseract-ocr tesseract-ocr-eng ffmpeg sox \
      ruby ruby-dev \
      openssl sagemath \
      openjdk-17-jdk-headless \
      sqlmap \
 && rm -rf /var/lib/apt/lists/*

# optional extras (universe packages; do not fail the build)
RUN apt-get update \
 && (apt-get install -y apktool nikto || echo "apktool/nikto unavailable; skipping") \
 && rm -rf /var/lib/apt/lists/*

# radare2 (prebuilt release)
ARG RADARE2_VERSION=5.8.8
RUN curl -fsSL -o /tmp/radare2.deb \
      "https://github.com/radareorg/radare2/releases/download/${RADARE2_VERSION}/radare2_${RADARE2_VERSION}_amd64.deb" \
 && apt-get install -y /tmp/radare2.deb \
 && rm -f /tmp/radare2.deb

# Ghidra + headless decompile wrapper
ARG GHIDRA_TAG=Ghidra_11.0.1_build
ARG GHIDRA_ZIP=ghidra_11.0.1_PUBLIC_20240130.zip
RUN curl -fsSL -o /tmp/ghidra.zip \
      "https://github.com/NationalSecurityAgency/ghidra/releases/download/${GHIDRA_TAG}/${GHIDRA_ZIP}" \
 && unzip -q /tmp/ghidra.zip -d /opt \
 && mv /opt/ghidra_* /opt/ghidra \
 && rm -f /tmp/ghidra.zip
COPY ghidra/DecompileAll.java /opt/ghidra/scripts/DecompileAll.java
COPY ghidra/ghidra-decompile.sh /usr/local/bin/ghidra-decompile
RUN chmod +x /usr/local/bin/ghidra-decompile

# jadx (Android decompiler)
ARG JADX_VERSION=1.4.7
RUN curl -fsSL -o /tmp/jadx.zip \
      "https://github.com/skylot/jadx/releases/download/v${JADX_VERSION}/jadx-${JADX_VERSION}.zip" \
 && unzip -q /tmp/jadx.zip -d /opt/jadx \
 && ln -s /opt/jadx/bin/jadx /usr/local/bin/jadx \
 && rm -f /tmp/jadx.zip

# flatter (fast lattice reduction)
RUN git clone --depth=1 https://github.com/keeganryan/flatter /tmp/flatter \
 && cmake -S /tmp/flatter -B /tmp/flatter/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /tmp/flatter/build -j"$(nproc)" \
 && cmake --install /tmp/flatter/build \
 && ldconfig \
 && rm -rf /tmp/flatter

# stegseek (build from source)
RUN apt-get update \
 && apt-get install -y libmhash-dev libmcrypt-dev \
 && rm -rf /var/lib/apt/lists/* \
 && git clone --depth=1 https://github.com/RickdeJager/stegseek /tmp/stegseek \
 && cmake -S /tmp/stegseek -B /tmp/stegseek/build \
 && cmake --build /tmp/stegseek/build --target install \
 && rm -rf /tmp/stegseek

# zsteg (ruby gem)
RUN gem install --no-document zsteg

# python CTF libraries
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
 && pip3 install --no-cache-dir \
      pwntools pycryptodome sympy gmpy2 requests Pillow \
      z3-solver pytesseract scipy numpy capstone unicorn \
      ropgadget fpylll flask volatility3 PyJWT tqdm angr

# RsaCtfTool
RUN git clone --depth=1 https://github.com/RsaCtfTool/RsaCtfTool /opt/RsaCtfTool \
 && pip3 install --no-cache-dir /opt/RsaCtfTool \
 && ln -sf /opt/RsaCtfTool/RsaCtfTool.py /usr/local/bin/RsaCtfTool

# CADO-NFS (heavy; WITH_HEAVY=0 to skip)
ARG WITH_HEAVY=1
RUN if [ "$WITH_HEAVY" = "1" ]; then \
      git clone --depth=1 https://gitlab.inria.fr/cado-nfs/cado-nfs.git /opt/cado-nfs \
   && cd /opt/cado-nfs && make -j"$(nproc)" \
   && printf '%s\n' \
        '#!/usr/bin/env bash' \
        'BUILD=/opt/cado-nfs/build' \
        'HOST=$(hostname)' \
        'if [ ! -d "$BUILD/$HOST" ]; then' \
        '  existing=$(ls -d "$BUILD"/* 2>/dev/null | head -n1)' \
        '  [ -n "$existing" ] && ln -sfn "$existing" "$BUILD/$HOST"' \
        'fi' \
        'exec python3 /opt/cado-nfs/cado-nfs.py "$@"' \
        > /usr/local/bin/cado-nfs \
   && chmod +x /usr/local/bin/cado-nfs; \
    fi

# PyTorch / Keras (optional; WITH_TORCH=1 to enable)
ARG WITH_TORCH=0
RUN if [ "$WITH_TORCH" = "1" ]; then \
      pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
   && pip3 install --no-cache-dir keras; \
    fi

# Podman (optional; WITH_PODMAN=1 to enable nested containers)
ARG WITH_PODMAN=0
RUN if [ "$WITH_PODMAN" = "1" ]; then \
      apt-get update \
   && apt-get install -y curl gnupg2 \
   && . /etc/os-release \
   && echo "deb https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/ /" \
        > /etc/apt/sources.list.d/kubic.list \
   && curl -fsSL "https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/Release.key" \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/kubic.gpg \
   && apt-get update \
   && apt-get install -y podman buildah fuse-overlayfs slirp4netns \
   && rm -rf /var/lib/apt/lists/*; \
    fi

# sudo (own layer so the heavy build cache above stays valid)
RUN apt-get update \
 && apt-get install -y sudo \
 && rm -rf /var/lib/apt/lists/*

# ARM tooling (own layer so the heavy build cache above stays valid)
RUN apt-get update \
 && apt-get install -y \
      qemu-user-static gdb-multiarch binutils-aarch64-linux-gnu gcc-aarch64-linux-gnu \
 && (apt-get install -y qemu-user-binfmt || echo "binfmt registration unavailable; skipping") \
 && rm -rf /var/lib/apt/lists/*

# non-root user
ARG USERNAME=ctfplayer
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid "$USER_GID" "$USERNAME" \
 && useradd --uid "$USER_UID" --gid "$USER_GID" -m -s /bin/bash "$USERNAME" \
 && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/$USERNAME" \
 && chmod 0440 "/etc/sudoers.d/$USERNAME" \
 && mkdir -p "/home/$USERNAME/ctf_files" "/home/$USERNAME/workspace" \
 && chown -R "$USER_UID:$USER_GID" "/home/$USERNAME"
COPY tools.txt /tools.txt

USER $USERNAME
WORKDIR /home/$USERNAME
CMD ["sleep", "infinity"]
