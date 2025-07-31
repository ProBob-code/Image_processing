# Base image: CentOS Stream 9 from Quay.io
FROM quay.io/centos/centos:stream9

# Set environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk
ENV SPARK_HOME=/opt/spark
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$PATH
ENV DOCROOT="/opt/airflow/api"

# Enable EPEL and RPM Fusion repositories
RUN dnf install -y epel-release && \
    dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-9.noarch.rpm && \
    dnf config-manager --set-enabled crb

# Install required packages
RUN dnf update -y && \
    dnf install -y gcc gcc-c++ make libffi-devel openssl-devel mysql-devel procps-ng tzdata \
    zlib-devel libjpeg-turbo-devel libpng-devel freetype-devel libwebp-devel libtiff-devel openjpeg2-devel \
    cmake pkgconfig gcc-gfortran gtk3-devel mesa-libGL-devel harfbuzz-devel libXrender-devel libSM-devel libXext-devel \
    git wget autoconf automake libtool nano openssh-server screen mediainfo ImageMagick ImageMagick-devel \
    ffmpeg ffmpeg-devel yasm nasm python3 python3-pip python3-virtualenv java-11-openjdk-devel cronie && \
    dnf clean all

# Install missing packages from source
RUN dnf groupinstall -y "Development Tools" && \
    # Install libogg
    curl -LO https://downloads.xiph.org/releases/ogg/libogg-1.3.5.tar.gz && \
    tar -xzf libogg-1.3.5.tar.gz && \
    cd libogg-1.3.5 && \
    ./configure --enable-shared && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf libogg-1.3.5* && \
    # Install x264
    git clone --depth 1 https://code.videolan.org/videolan/x264.git && \
    cd x264 && \
    ./configure --enable-static --enable-shared && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf x264 && \
    # Install x265
    git clone --depth 1 https://github.com/videolan/x265.git && \
    cd x265/build/linux && \
    cmake ../../source && \
    make -j$(nproc) && \
    make install && \
    cd ../../../ && \
    rm -rf x265 && \
    # Install libvpx
    git clone --depth 1 https://chromium.googlesource.com/webm/libvpx && \
    cd libvpx && \
    ./configure --enable-runtime-cpu-detect --enable-vp8 --enable-vp9 --enable-shared && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf libvpx && \
    # Install lame
    curl -LO https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz && \
    tar -xzf lame-3.100.tar.gz && \
    cd lame-3.100 && \
    ./configure --enable-shared && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf lame-3.100* && \
    # Install opus
    curl -LO https://archive.mozilla.org/pub/opus/opus-1.3.1.tar.gz && \
    tar -xzf opus-1.3.1.tar.gz && \
    cd opus-1.3.1 && \
    ./configure --enable-shared && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf opus-1.3.1* && \
    # Install libtheora
    curl -LO http://downloads.xiph.org/releases/theora/libtheora-1.1.1.tar.gz && \
    tar -xzf libtheora-1.1.1.tar.gz && \
    cd libtheora-1.1.1 && \
    ./configure --enable-shared --disable-examples && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf libtheora-1.1.1* && \
    # Install libvorbis
    curl -LO https://downloads.xiph.org/releases/vorbis/libvorbis-1.3.7.tar.gz && \
    tar -xzf libvorbis-1.3.7.tar.gz && \
    cd libvorbis-1.3.7 && \
    ./configure --enable-shared --disable-examples && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf libvorbis-1.3.7*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set timezone
RUN ln -fs /usr/share/zoneinfo/Asia/Kolkata /etc/localtime && \
    dnf reinstall -y tzdata && \
    dnf clean all

# Create directories
RUN mkdir -p /opt/airflow/api /opt/spark/jars /opt/airflow/models /opt/airflow/dags

# Copy application files
WORKDIR ${DOCROOT}
COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt && \
#     rm -rf /opt/venv/lib/python3.9/site-packages/nvidia/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.1+cpu torchvision==0.17.1+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /opt/venv/lib/python3.9/site-packages/nvidia/*

# Copy application files
COPY api/ ${DOCROOT}/
COPY models/ /opt/airflow/models/
COPY dags/ /opt/airflow/dags/

# copy cron entry
COPY crontab /var/spool/cron/root

COPY api/start.sh start.sh
COPY api/run_script.sh run_script.sh

# Set permissions
RUN chmod +x start.sh run_script.sh && \
    chown root:root /var/spool/cron/root && \
    chmod 0644 /var/spool/cron/root && \
    rm -rf ${DOCROOT}/__pycache__ ${DOCROOT}/api/__pycache__ ${DOCROOT}/api/.git

# Expose port and start services
EXPOSE 8081
CMD ["sh", "start.sh"]
# CMD /usr/sbin/sshd -D & ulimit -c 0 && gunicorn --workers 5 app:app -b 0.0.0.0:8089 --timeout 3000
