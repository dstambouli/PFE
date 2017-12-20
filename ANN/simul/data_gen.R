set.seed(8)

w1 = 0.5*matrix(rnorm(12),3,4)
w2 = 0.5*matrix(rnorm(4),4,1)

N = 1000
x1 = rnorm(N, 0, 0.25)
x2 = rnorm(N, 0.5, 0.30)
b = rep(1, N)
epsilon = rnorm(N, 0, 1)


input = matrix(c(b, x1,x2), nrow = length(b))

sig = function(x){
    return ((1+exp(-x))^(-1))
}
Y = matrix(nrow = length(b), ncol = 1, 0)
#for (i in 1:nrow(input)) {
#    temp1 = sig(input[i,1:3]%*%w1)
#    temp2 = sig(temp1%*%w2)
#    Y[i] = temp2
#}

n1 = 0
n2 = 0
for (i in 1:nrow(input)) {
  temp1 = sig(input[i,1:3]%*%w1)
  temp2 = sig(temp1%*%w2) + epsilon[i]
  if(temp2 > 0.45){
    Y[i] = 1 
    n1 = n1 +1
  }
  if(temp2 <= 0.45){
    Y[i] = 0
    n2 = n2 + 1
  }
}



filename <- file('set1.csv', open = "w")
writeLines(paste(c("b", "x1", "x2", "Y"), collapse = ";"), filename)
for(i in 1:nrow(input)){
  writeLines(paste(c(input[i,1:3], Y[i]), collapse = ";"), filename)
}

close(filename)

