# Privacy Bias 



## Instructions

<details><summary>Using a locally built Docker image</summary>

1. Build the Docker image:

`docker build -t privacy_bias:latest .`

2. Run

`docker run -p 8888:8888 \
    -v ./plots/objects:/home/auditor/./plots/objects \
    -v ./plots:/home/auditor/plots \
    privacy_bias:latest`

3. Open your browser and go to: [http://localhost:8888](http://localhost:8888
)

</details>


## Using VScode 

<details>

1. Install the Dev Containers extension in VS Code.

2. Open the repository in VS Code.

3. Press F1, choose Dev Containers: Open Folder in Container…, and select the project folder.

</details>


## Important files

plots/plots.ipynb : Jupyter notebook to generate all the papers figures.


 
