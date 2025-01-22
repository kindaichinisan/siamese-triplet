FROM harbor.ai-stack.dso/gx/ngc-pytorch-codeserver-screen:22.03-py3

WORKDIR /home/hwenjun

# Add the certificate to the trust store
RUN wget --no-check-certificate https://gitlab.ai-stack.dso/eugene/ai-stack-certificates/-/raw/main/dso/DSOCA50.pem?inline=false -O DSOCA50.pem
RUN wget --no-check-certificate https://gitlab.ai-stack.dso/eugene/ai-stack-certificates/-/raw/main/dso/DSOVisibilityCA.pem?inline=false -O DSOVisibilityCA.pem
RUN apt-get update && apt-get -y install ca-certificates
RUN mkdir -p /usr/local/share/ca-certificates
RUN cp DSOCA50.pem /usr/local/share/ca-certificates/DSOCA50.crt
RUN cp DSOVisibilityCA.pem /usr/local/share/ca-certificates/DSOVisibilityCA.crt
RUN update-ca-certificates

#git (Ubuntu)
RUN wget --no-check-certificate https://gitlab.ai-stack.dso/eugene/ai-stack-certificates/-/raw/main/dso/certbundle.zip?inline=false -O certbundle.zip
RUN unzip certbundle.zip
RUN git config --global http.sslcainfo $(pwd)/certbundle.crt

#Python pip (Ubuntu)
RUN pip config set global.cert $(pwd)/certbundle.crt
RUN pip3 config set global.cert $(pwd)/certbundle.crt

#urllib
RUN echo "export SSL_CERT_FILE=/path/to/certbundle.crt" >> ~/.bashrc
RUN source ~/.bashrc

RUN apt-get update && apt-get install -y git


RUN git clone https://github.com/kindaichinisan/siamese-triplet.git

WORKDIR /home/hwenjun/siamese-triplet

#install python requirements
#RUN apt install python=3.11
#RUN pip install -r requirements.txt
