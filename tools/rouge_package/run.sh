cp ROUGE-1.5.5.tar.gz ~
tar -xvf ~/ROUGE-1.5.5.tar.gz --directory ~
rm ~/ROUGE-1.5.5.tar.gz
sudo apt install libxml-parser-perl
sudo pip3 install pyrouge

cd ~
mkdir .pyrouge
echo [pyrouge settings] >> .pyrouge/settings.ini
echo home_dir = $PWD/ROUGE-1.5.5 >> .pyrouge/settings.ini

cd ~/ROUGE-1.5.5/data/
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
