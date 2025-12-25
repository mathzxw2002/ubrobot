

https://github.com/mathzxw2002/VideoChat

forked from: https://github.com/Henry-23/VideoChat





## Bug Fix

### 0, ModuleNotFoundError: No module named 'mmcv._ext' 
mmcv

python3 setup.py build_ext --inplace

sudo MMCV_WITH_OPS=1 pip install -e .  --break-system-packages



mmcv 2.1.0
mmdet 3.3.0
mmpose 1.3.2


### 1, TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
pip install httpx==0.27.2
https://blog.csdn.net/weixin_44003104/article/details/144375184


###
  File "/home/sany/.local/lib/python3.12/site-packages/gradio_client/utils.py", line 880, in get_type
    if "const" in schema:
       ^^^^^^^^^^^^^^^^^
TypeError: argument of type 'bool' is not iterable

#pip install gradio==6.2.0 gradio_client==2.0.2


gradio-5.4.0 gradio-client-1.4.2



###

pip install numpy==1.26.4 scipy==1.11.4 librosa==0.10.1 opencv-python==4.8.1.78  --break-system-packages


pip install huggingface-hub==0.25.1 --break-system-packages


tokenizers 0.19.1 requires huggingface-hub<1.0,>=0.16.4, but you have huggingface-hub 1.2.3 which is incompatible.
transformers 4.44.1 requires huggingface-hub<1.0,>=0.23.2, but you have huggingface-hub 1.2.3 which is incompatible.


## gradio启动 报错 TypeError: argument of type ‘bool‘ is not iterable
https://blog.csdn.net/qq_63234089/article/details/146914002



# Generate Encrypt
penssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

