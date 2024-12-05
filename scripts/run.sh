docker run -it --rm \
        -p 8000:8000 \
        -v "$PWD/models:/models:ro" \
        --platform linux/amd64 \
        --name model_backend model_backend-img \
        /bin/bash  -c 'for req in /models/*/model/*.txt; do pip install -r $req; done; python /src/api.py'
