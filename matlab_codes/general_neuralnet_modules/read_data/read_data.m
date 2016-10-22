% Purpose: Read data

sl            = 100;
train_numbats = 100;
val_numbats   = 5;
test_numbats  = 5;

switch cfn
    case 'ls'
        dout            = 1;
        train_batchdata = randn(sl,dout,train_numbats);
        val_batchdata   = randn(sl,dout,val_numbats);
        test_batchdata  = randn(sl,dout,test_numbats);
        
    case 'nll'
        dout            = 5;
        
        
        train_batchdata = zeros(sl,dout,train_numbats);
        val_batchdata   = zeros(sl,dout,val_numbats);
        test_batchdata  = zeros(sl,dout,test_numbats);
        
        for i = 1:train_numbats
            for j = 1:sl
                train_batchdata(j,randi(5),i) = 1;
            end
        end
        
end

