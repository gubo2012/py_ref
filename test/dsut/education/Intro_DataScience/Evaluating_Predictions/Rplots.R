set.seed(10)

x <- 1:10
y <- 0.5*x + rnorm(length(x))
train_dat <- data.frame(x,y)
#Fit for high level polynomial
lm_hp1 <- lm(y ~ x, data=train_dat)
lm_hp6 <- lm(y ~ poly(x,6), data=train_dat)
lm_hp9 <- lm(y ~ poly(x,9), data=train_dat)

lin_dat <- data.frame(x=seq(0,10,by=0.1))

plot(x, y, pch=21, bg='grey', cex=2, xaxt='n', yaxt='n', ylim=c(0,5), 
     main='9th Order Polynomial')
#lines(lin_dat$x, predict(lm_hp1, newdat=lin_dat), col='red')
lines(lin_dat$x, predict(lm_hp9, newdat=lin_dat), col='blue')
points(x, y, pch=21, bg='grey', cex=2)
axis(1, at=x)
axis(2, at=0:5, las=1)

#legend(1, 4.5, legend=c('6th order polynomial','9th order polynomial'), 
       lty=1, cex=0.9, col=c("red","blue"))


#####################

x <- runif(100,0,10)
y <- 0.5*x + rnorm(length(x))
test_dat <- data.frame(x,y)

plot(x, y, bg='grey', cex=0.5, xaxt='n', yaxt='n', ylim=c(-2,6), 
     main='Test predictions out of sample')
lines(lin_dat$x, predict(lm_hp1, newdat=lin_dat), col='red')
lines(lin_dat$x, predict(lm_hp9, newdat=lin_dat), col='blue')
#points(x, y, pch=21, bg='grey', cex=2)
axis(1, at=0:10)
axis(2, at=-2:6, las=1)
legend(0, 6.2, legend=c('linear regression','9th order polynomial'), 
       lty=1, cex=0.9, col=c("red","blue"))

#Overlapping histogram

resid1 <- test_dat$y - predict(lm_hp1, newdat=test_dat)
resid2 <- test_dat$y - predict(lm_hp9, newdat=test_dat)

resid_dat <- data.frame(error_lm=resid1, error_9th=resid2)

boxplot(

hist(resid2, col='blue', breaks=10, xlab='Error (Residual)')
hist(resid1, col='red', add=T)

mean(resid1)
