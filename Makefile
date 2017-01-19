.PHONY: build clean test deploy

export MVNFLAGS=-Djava.io.tmpdir=/nail/tmp
export MAVEN := mvn3
export JAVA_HOME := /usr/lib/jvm/java-8-oracle-1.8.0.45

clean:
	$(MAVEN) clean

build:
	$(MAVEN) install -U -DskipTests

deploy:
	$(MAVEN) -N deploy -DskipTests

test:
	$(MAVEN) $(MVNFLAGS) test
