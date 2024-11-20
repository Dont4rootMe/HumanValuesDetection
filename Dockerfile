FROM pytorch/pytorch AS backend


# install server packages
RUN python -m pip install --upgrade pip
RUN pip install \
    uvicorn \
    fastapi \
    pydentic

# preinstall comon dependencies
RUN pip install torch==2.2.1 \
    transformers==4.43.4 \
    boto3==1.35.55 \
    gliner==0.1.12 \
    numpy

# run server
COPY ./src /src
RUN mkdir /models

EXPOSE 8000