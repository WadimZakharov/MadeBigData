name := "cos_dist"

version := "0.1"

scalaVersion := "2.12.12"

val spartVersion = "3.0.1"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % spartVersion withSources(),
  "org.apache.spark" %% "spark-mllib" % spartVersion withSources(),
)
