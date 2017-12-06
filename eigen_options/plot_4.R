library(readr);
library(dplyr);
library(ggplot2);
library(reshape2);

readthecsv <- function(filename) {
  df <- read_csv(filename, col_names = FALSE, col_types = cols(X501 = col_skip()));
  colnames(df) <- seq(1,500);
  df <- t(apply(df, 1, cumsum))
  melt(df, value.name='reward') %>% rename(run=Var1, episode=Var2) %>% mutate(avg_reward=reward/episode) %>% mutate(name=sub('qlearner-([[:digit:]]+) options.csv', '\\1 options', filename))
}

df <- readthecsv('primitive.csv') %>% mutate(name='primitive')
for (file in list.files(pattern="*.csv")) {
  if (file != 'primitive.csv') {
    df <- full_join(df, readthecsv(file))
  }
}
ggplot(data=df, aes(x=episode, y=avg_reward, colour=name)) + stat_smooth(method="loess", span=0.1, se=TRUE, aes(fill=name), alpha=0.3)
