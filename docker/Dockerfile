FROM dolfinx/dolfinx:nightly
RUN python3 -m pip install -U pip
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN python -m pip install --index-url https://test.pypi.org/simple/ --no-deps phiFEM==0.4.0
ENV PYTHONPATH="$PYTHONPATH:/home/dolfinx/shared/"
