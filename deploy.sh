pyinstaller --hidden-import cloudpathlib \
            --hidden-import importlib.metadata \
            --hidden-import importlib_metadata \
            --collect-all cloudpathlib \
            cryofold.py
dist=/data/zhangqflab/cryofold/
rm -rf $dist
mv dist/cryofold /data/zhangqflab/
# cp -r /data/zhangqflab/sw/cpx $dist/_internal/
echo "copying weight"
cp -r /data/zhangqflab/models/esm3/data $dist
cp -r deploy/* $dist
cp -r misc $dist
mv $dist/misc/cryonet $dist/_internal/
cp -r /home/zhangqflab/usr/miniconda3/envs/cryofold/lib/python3.12/site-packages/biotite $dist/_internal/


cd /data/zhangqflab/
echo "compressing package..."

tar -zcf cryofold_standalone.tar.gz cryofold
rsync -avP /data/zhangqflab/cryofold_standalone.tar.gz jsptgpu@20.20.91.11:/data01/share/
