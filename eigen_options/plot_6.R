library(readr);
library(dplyr);
library(ggplot2);
library(reshape2);

avg_step <- function(filename) {
  df <- read_csv(filename, col_names = FALSE, col_types = cols(X901 = col_skip()));
  mean(apply(df, 1, function(x) min(which(x == 1))))
}

y <- c()
for (x in 4:350) {
  y <- c(y, avg_step(sprintf("qlearner-%d options.csv", x)))
}
df <- melt(y)
df$x <- 4:350
ggplot(data=df, aes(x=x, y=value)) + geom_point() + geom_line() + geom_line(y=600)
