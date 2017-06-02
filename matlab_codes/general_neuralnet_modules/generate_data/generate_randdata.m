% generate random data for testing (numerical gradient matching)

train_numbats = 1;
val_numbats = 2;
test_numbats = 2;

[train_batchdata,train_batchtargets,train_clv] = gen_randdata(train_numbats,din,dout,sl,ol_type);
[val_batchdata,val_batchtargets,val_clv] = gen_randdata(val_numbats,din,dout,sl,ol_type);
[test_batchdata,test_batchtargets,test_clv] = gen_randdata(test_numbats,din,dout,sl,ol_type);

load('a.mat','train_batchdata','train_batchtargets','train_clv')