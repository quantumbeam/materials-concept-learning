FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Install additional packages
RUN pip install --no-cache-dir ninja yacs matplotlib \
    sqlalchemy pymysql pyyaml scikit-learn jupyterlab ipywidgets hydra ase pytorch-lightning \
&& mkdir ~/.jupyter
ADD jupyter_notebook_config.py /root/.jupyter/

RUN conda install pyg -c pyg -y
RUN conda install --channel conda-forge nglview -y\
&& conda install -c pytorch faiss-gpu -y\
&& conda clean -ya

# workaround for version related issue
RUN pip install --no-cache-dir nbconvert==6.4.4 pymatgen==2022.5.26 tensorboard