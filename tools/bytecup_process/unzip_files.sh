unzip ../Bytecup2018.zip
flead="bytecup.corpus.train."
ftail=".zip"
for i in {1..8}
  do unzip $flead$i$ftail
  done
rm *.zip