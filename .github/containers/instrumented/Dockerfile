# I pin the multi-platform Ubuntu 24.04 image and every apt repository in time.
FROM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac
# My minimal base has no CA bundle. I verify this public bootstrap archive before
# extracting its certificates; TLS and apt signature checks remain enabled.
ADD https://snapshot.ubuntu.com/ubuntu/20261019T000000Z/pool/main/c/ca-certificates/ca-certificates_20240203_all.deb /tmp/ca-certificates.deb
RUN echo "641de77d8f142cfd62a1a6f964ba67b20754d3337c480efb529d086075a06c9a  /tmp/ca-certificates.deb" | sha256sum -c - \
    && dpkg-deb -x /tmp/ca-certificates.deb / \
    && mkdir -p /etc/ssl/certs \
    && cat /usr/share/ca-certificates/mozilla/*.crt > /etc/ssl/certs/ca-certificates.crt \
    && rm /tmp/ca-certificates.deb
COPY ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources
RUN rm -f /etc/apt/sources.list \
    && apt-get -o Acquire::Retries=2 -o APT::Update::Error-Mode=any update \
    && DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=2 install -y --no-install-recommends \
       build-essential git ca-certificates python3 python3-yaml pkg-config \
       libssl-dev libffi-dev libsqlite3-dev lcov bc \
       libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libutf8proc-dev \
       libncurses-dev libreadline-dev libevent-dev libuv1-dev libbullet-dev \
       libglfw3-dev libglew-dev freeglut3-dev gforth \
    && dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' > /etc/nanolang-ci-packages \
    && git config --system --add safe.directory /workspace \
    && rm -rf /var/lib/apt/lists/*
# I retain compiler alternatives and process diagnostics previously supplied by
# the outer hosted image, and the hosted runner's numeric account identity.
RUN apt-get -o Acquire::Retries=2 -o APT::Update::Error-Mode=any update \
    && DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=2 install -y --no-install-recommends clang procps passwd \
    && useradd --uid 1001 --create-home --shell /bin/bash runner \
    && dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' > /etc/nanolang-ci-packages \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
WORKDIR /workspace
