library(ggplot2)
library(patchwork)

data=read.table("../data/billboard.subcharts.yearly.data", head=FALSE, sep="\t")
rb_data=data[data$V1=="rb",]
country_data=data[data$V1=="country",]
rock_data=data[data$V1=="rock",]
pop_data=data[data$V1=="pop-airplay",]
rap_data=data[data$V1=="rap",]
rbhiphop_data=data[data$V1=="rbhiphop",]

big_text_theme <- theme(
  axis.title = element_text(size = 20),
  axis.text = element_text(size = 16),
  plot.title = element_text(size = 24),
  strip.text = element_text(size = 18)
)

scale <- scale_x_continuous(breaks = c(2005, 2015, 2025))
# scale_x_continuous(breaks = seq(2002, 2024, by = 9)) 

country_g=ggplot(country_data, aes(x=V3, y=V2)) + geom_point(alpha=10)+ ylab("") + xlab("country") + geom_ribbon(aes(ymin=V4, ymax=V5), alpha=0.2, fill="blue") + coord_cartesian(xlim = c(2002, 2025.5)) +ylim(0,5)+ scale + big_text_theme

rock_g=ggplot(rock_data, aes(x=V3, y=V2)) + geom_point(alpha=10)+ ylab("narrativity") + xlab("rock") + geom_ribbon(aes(ymin=V4, ymax=V5), alpha=0.2, fill="blue") + ylim(0,5)+ coord_cartesian(xlim = c(2002, 2025.5)) + scale + big_text_theme

pop_g=ggplot(pop_data, aes(x=V3, y=V2)) + geom_point(alpha=10)+ ylab("") + xlab("pop") + geom_ribbon(aes(ymin=V4, ymax=V5), alpha=0.2, fill="blue") + ylim(0,5)+ coord_cartesian(xlim = c(2002, 2025.5)) + scale + big_text_theme

rap_g=ggplot(rap_data, aes(x=V3, y=V2)) + geom_point(alpha=10)+ ylab("") + xlab("rap") + geom_ribbon(aes(ymin=V4, ymax=V5), alpha=0.2, fill="blue") + ylim(0,5)+ coord_cartesian(xlim = c(2002, 2025.5)) + scale + big_text_theme

rbhiphop_g=ggplot(rbhiphop_data, aes(x=V3, y=V2)) + geom_point(alpha=10)+ ylab("") + xlab("r&b/hip-hop") + geom_ribbon(aes(ymin=V4, ymax=V5), alpha=0.2, fill="blue") + ylim(0,5) + coord_cartesian(xlim = c(2002, 2025.5)) + scale + big_text_theme

pdf("billboard_subcharts_year.pdf", width=20)

(rock_g + pop_g + country_g + rbhiphop_g + rap_g) +
  plot_layout(ncol = 5, widths = rep(.5, 5)) &
  theme(plot.margin = margin(10, 30, 10, 0)) 

dev.off()