#!/bin/bash
USER_ID=${LOCAL_USER_ID:-9001}
USER = ${markpp}
echo "Starting with username : ${USER} and UID : ${USER_ID}"
useradd -s /bin/bash -u $USER_ID -o -p 1234 #-m $USER create home dir
export HOME=/home/markpp
su $USER bash -c 'bash'
