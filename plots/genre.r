library(ggplot2)
library(patchwork)

big_text_theme <- theme(
  axis.title = element_text(size = 20),
  axis.text = element_text(size = 16),
  plot.title = element_text(size = 24),
  strip.text = element_text(size = 18)
)

scale <- scale_x_continuous(breaks = c(1960, 1970, 1980, 1990, 2000, 2010, 2015, 2025))

data=read.table("../data/hiphop_rap_country.proportion.txt")
country=data[data$V1 == "country", ]
country_g=ggplot(country, aes(x=V2, y=V3)) + geom_point() + geom_ribbon(aes(ymin=V4, ymax=V5), alpha=0.2, fill="blue") + ylim(0,1) + xlab("") + ylab("% country") + big_text_theme# + scale

data=read.table("../data/hiphop_rap_country.proportion.txt")
hiphoprap=data[data$V1 == "hiphop", ]
hiphop_g=ggplot(hiphoprap, aes(x=V2, y=V3)) + geom_point() + geom_ribbon(aes(ymin=V4, ymax=V5), alpha=0.2, fill="blue") + ylim(0,1) + xlab("") + ylab("% hip hop/rap") + big_text_theme #+ scale

pdf("hip_hop_country.pdf", width=14)

(hiphop_g+ country_g) +
  plot_layout(ncol = 2, widths = rep(1,2)) &
  theme(plot.margin = margin(10, 30, 10, 0)) 
 dev.off()
