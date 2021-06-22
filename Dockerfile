# FROM:ベースとなるDockerimageの指定．詳しくはDockerHubのBase Imagesを参照．
FROM python:3.6.9


ENV PROGRAM_DIR=/opt/ml/code
RUN pip install --upgrade pip
RUN apt-get update
RUN pip install sagemaker-training
# /opt/ml/code ディレクトリの作成
RUN mkdir -p $PROGRAM_DIR
#作業ディレクトリの設定
WORKDIR $PROGRAM_DIR
#作業ディレクトリにrequirements.txtのコピー
COPY requirements.txt $PROGRAM_DIR/requirements.txt
#train.pyを移動
COPY train.py  $PROGRAM_DIR/train.py
COPY train.py  $SAGEMAKER_CODE_DIR/train.py
#train.pyで必要パッケージをインストール
RUN pip install -r requirements.txt 
#永続的な環境変数を設定
ENV SAGEMAKER_PROGRAM train.py 