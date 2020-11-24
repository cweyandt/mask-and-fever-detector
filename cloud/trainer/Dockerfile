FROM tensorflow/tensorflow:2.4.0rc2-gpu

# Update and install packages
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update --fix-missing && \
    apt-get install -y automake autotools-dev g++ git libcurl4-gnutls-dev wget libfuse-dev libssl-dev libxml2-dev make pkg-config
    
# Clone and run s3fs-fuse
RUN git clone https://github.com/s3fs-fuse/s3fs-fuse.git /tmp/s3fs-fuse && \
    cd /tmp/s3fs-fuse && ./autogen.sh && ./configure && make && make install && ldconfig && /usr/local/bin/s3fs --version

# Install python & packages
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y python3 python3-pip mosquitto-clients python3-opencv vim
RUN pip3 install paho-mqtt numpy sklearn pandas

# Remove packages
#RUN DEBIAN_FRONTEND=noninteractive apt-get purge -y wget automake autotools-dev g++ git make  && \
#    apt-get -y autoremove --purge && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set user and group
ENV USER='appuser'
ENV GROUP='appuser'
ENV UID='1000'
ENV GID='1000'

RUN groupadd -g $GID $GROUP && \
    useradd -u $UID -g $GROUP -s /bin/sh -m $USER

# Install fuse
RUN apt-get update   && \
    apt install fuse && \
    chown ${USER}.${GROUP} /usr/local/bin/s3fs

# Config fuse
RUN chmod a+r /etc/fuse.conf && \
    perl -i -pe 's/#user_allow_other/user_allow_other/g' /etc/fuse.conf

# Create mnt point
ENV MNT_POINT_PATH=/bucket
RUN mkdir -p $MNT_POINT_PATH && \
    chmod g+w $MNT_POINT_PATH && chown $USER $MNT_POINT_PATH

# Switch to user
USER ${UID}:${GID}

# copy file
COPY ./scripts/* /home/${USER}/

# Execute
ENV S3_BUCKET='covid19-detector'
WORKDIR /home/${USER}
CMD /usr/local/bin/s3fs $S3_BUCKET $MNT_POINT_PATH -o iam_role=auto -o allow_otherl; python3 dummy.py

