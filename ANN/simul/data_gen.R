set.seed(8)

w1 = matrix(rnorm(6),2,3)
w2 = 0.5*matrix(rnorm(3),3,1)

N = 1000
x1 = rnorm(N, 0, 0.5)
x2 = rnorm(N, 0.5, 0.30)
epsilon = rnorm(N, 0, 0.01)


input = matrix(c(x1,x2), nrow = N)

sig = function(x){
    return ((1+exp(-x))^(-1))
}
Y = matrix(nrow = N, ncol = 1, 0)


for (i in 1:nrow(input)) {
  temp1 = sig(input[i,1:2]%*%w1)
  Y[i] = sig(temp1%*%w2) + epsilon[i]

}



filename <- file('set1.csv', open = "w")
writeLines(paste(c( "x1", "x2", "Y"), collapse = ";"), filename)
for(i in 1:nrow(input)){
  writeLines(paste(c(input[i,1:2], Y[i]), collapse = ";"), filename)
}

close(filename)

