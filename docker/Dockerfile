FROM dolfinx/dolfinx:stable
RUN python3 -m pip install -U pip
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
ENV PYTHONPATH="$PYTHONPATH:/home/dolfinx/shared/"
ENV MYPYPATH="$MYPYPATH:/home/dolfinx/shared/phiFEM/stubs"