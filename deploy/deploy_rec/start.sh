#!/bin/bash

NUM_PROCESS=1
PORT=8089

getpid()
{
    #PID=$(pgrep -f "gunicorn -b 0.0.0.0:"$PORT" -w "$NUM_PROCESS" img_server_textline:app")
    PID=$(ps -elf | grep gunicorn |grep img_server_textrec:app | awk '{print $4}')
    if [ "x$?" != "x0" ]
    then
        PID=''
    fi
    echo $PID
}

start_service()
{
    echo "start..."
    PID=$(getpid)
    if [ "x$PID" != "x" ]
    then
        echo "pid : "$PID" is running"
        exit 0
    fi
    filepath=$(cd "$(dirname "$0")";pwd)
    cd $filepath
    nohup gunicorn -b 0.0.0.0:$PORT -w $NUM_PROCESS -t 300 img_server_textrec:app &> /dev/null < /dev/null &
}

stop_service()
{
    echo "stop ..."
    PID=$(getpid)
    while [ "x$PID" != "x" ]
    do
        for p in $PID;
        do 
            kill $p
        done
        sleep 1
        PID=$(getpid)
    done   
}


case "$1" in 
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        echo "restart"
        stop_service
        start_service
        ;;
    status)
        getpid
        ;;
esac




