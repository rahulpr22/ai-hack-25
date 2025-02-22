build:
	docker build -t agent .

push:
	docker tag agent:latest karthik1900/devops-1:sales
	docker push karthik1900/devops-1:sales

run:
	docker run -p 10000:10000 agent