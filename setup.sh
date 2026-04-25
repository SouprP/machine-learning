export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

for dir in $(find $VIRTUAL_ENV/lib/ -type d -name "lib" | grep nvidia); do
    export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
done
