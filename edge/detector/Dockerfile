# Base image
FROM drminix/detector

# # Create working directory
RUN mkdir -p /usr/src/app

# Copy contents
COPY . /usr/src/app
WORKDIR /usr/src/app/

# install the other python packages
RUN pip3 install -r requirements.txt
CMD ["python3", "maskDetector.py"]
