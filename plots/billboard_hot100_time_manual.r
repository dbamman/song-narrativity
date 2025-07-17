library(ggplot2)

data=read.table("../data/billboard.hot100_manual.yearly.data")
pdf("billboard_hot100_manual_year.pdf", width=7)
ggplot(data, aes(x=V2, y=V1)) + geom_point() + geom_ribbon(aes(ymin=V3, ymax=V4), fill="blue", alpha=0.3) + xlim(1960,2025) + xlab("") + ylab("narrativity")
dev.off()
