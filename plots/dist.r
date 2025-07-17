library(ggplot2)
library(patchwork)

data=read.table("../data/annotations.tsv", sep="\t", quote = "\"", head=TRUE)

pdf("annotation_dist.pdf", width=18)

agent_g=ggplot(data, aes(x=Final.AGENT)) + geom_density(fill = "blue", alpha = 0.3) + xlab("Agents") + ylab("density") + ylim(0,.4)
event_g=ggplot(data, aes(x=Final.EVENTS)) + geom_density(fill = "blue", alpha = 0.3) + xlab("Events") + ylab("")+ ylim(0,.4)
world_g=ggplot(data, aes(x=Final.WORLD)) + geom_density(fill = "blue", alpha = 0.3) + xlab("World") + ylab("")+ ylim(0,.4)

(agent_g + event_g + world_g) +
  plot_layout(ncol = 3, widths = rep(1, 3)) &
  theme(plot.margin = margin(10, 10, 10, 10)) 


dev.off()