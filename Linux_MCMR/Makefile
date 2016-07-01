all:
	wget -nc http://soplex.zib.de/download/release/soplex-1.7.2.tgz
	tar zxvf soplex-1.7.2.tgz
	$(MAKE) -C soplex-1.7.2 all
	g++ Train.cpp -L soplex-1.7.2/lib/ -lsoplex -lz -w -o mcmrtrain
	g++ Predict.cpp -L soplex-1.7.2/lib/ -lsoplex -lz -w -o mcmrpredict
	echo 'export PATH=$$PATH:.' >> ~/.bashrc

clean:
	rm -rf mcmrpredict mcmrtrain soplex-1.7.2 soplex-1.7.2.tgz*
