w1 = 0.5*matrix(rnorm(12),3,4)
w2 = 0.5*matrix(rnorm(4),4,1)

x1 = rnorm(5000, 0, 1)
x2 = rnorm(5000, 0.5, 0.30)
b = rep(1, 5000)

plot(x1)
points(x2, col=2)

input = matrix(c(b, x1,x2), nrow = length(b))

sig = function(x){
    return ((1+exp(-x))^(-1))
}
Y = matrix(nrow = length(b), ncol = 1, 0)
n1 = 0
n2 = 0
for (i in 1:nrow(input)) {
  temp1 = sig(input[i,1:3]%*%w1)
  temp2 = sig(temp1%*%w2)
  if(temp2 > 0.515){
    Y[i] = 1 
    n1 = n1 +1
  }
  if(temp2 <= 0.515){
    Y[i] = 0
    n2 = n2 + 1
  }
  
}



filename <- file('data.csv', open = "w")
for(i in 1:nrow(input)){
  writeLines(paste(c(input[i,1:3], Y[i]), collapse = ";"), filename)
}

close(filename)

