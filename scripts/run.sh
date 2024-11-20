helpFunction()
{
   echo ""
   echo "Usage: $0 -m path to model"
   echo -e "\t-m determine path to modules with model apis"
   exit 1 # Exit script after printing help
}

while getopts "d:m:" opt
do
   case "$opt" in
      m ) model_api="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

docker run -it --rm \
        -p 8000:8000 \
        -v "$model_api:/models:ro" \
        --platform linux/amd64 \
        --name model_backend model_backend-img \
        /bin/bash  -c 'for req in /models/*/model/*.txt; do pip install -r $req; done; python /src/api.py'
