set.seed(8)

w1 = matrix(rnorm(6, mean=0, sd=2),2,3)
w2 = matrix(rnorm(3, mean=0, sd=1),3,1)
b2 = rnorm(1, mean=0, sd=1)
N = 5000
x1 = rnorm(N, 9, 9)
x2 = rnorm(N, 8, 1)
epsilon = rnorm(N, 0, 0.01)


input = matrix(c(x1,x2), nrow = N)

sig = function(x){
    return ((1+exp(-x))^(-1))
}
Y = matrix(nrow = N, ncol = 1, 0)


for (i in 1:nrow(input)) {
  temp1 = input[i,1:2]%*%w1 
  temp1 = sig(temp1)
  temp2 = temp1%*%w2 + b2
  Y[i] = sig(temp2) + epsilon[i]

}



filename <- file('set1.csv', open = "w")
writeLines(paste(c( "x1", "x2", "Y"), collapse = ";"), filename)
for(i in 1:nrow(input)){
  writeLines(paste(c(input[i,1:2], Y[i]), collapse = ";"), filename)
}

close(filename)

