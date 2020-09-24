FROM python3.6

RUN mkdir codes

COPY . codes/

RUN pip3 install -r codes/requirements.txt