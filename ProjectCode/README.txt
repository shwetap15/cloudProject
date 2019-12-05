We have executed the project code on the AWS EMR cluster. We have uploaded the following files - 
1. Flights.py - it contains all the source code for the custom-built Logistic Regression algorithm, Logistic Regression using MLib, and Random Forest algorithm using MLib 
2. DataAnalysis.py - it contains code for analyzing 15000 random samples in our dataset
3. PlotResults.py - We have downloaded Predicted and actual label values for custom-built logistic regression algorithm and this file contains code for plotting graphs of results
4. True_pred.txt - this is the output file of the Logistic regression algorithm executed for 15000 records for 500 iterations
5. True_pred100k.txt - this is the output file of the Logistic regression algorithm executed for 100000 records for 100 iterations


For running code on AWS cluster, we have executed the following steps- 
1. Created an S3 bucket on AWS and uploaded source Flights.py file and datasets.
2. Created SparkCluster on AWS console with required number of m5.xlarge instances
3. SSH to console and downloaded source file and dataset file on the console from S3 bucket
4. Executed following commands for downloading files from S3 bucket to local AWS console - 
aws s3 cp s3://BUCKET_NAME/Flights.py . 
aws s3 cp s3://BUCKET_NAME/randomData100k.csv . 
5. hadoop fs -put randomData100k.csv /input/
6. For running code on AWS - 
/usr/bin/spark-submit Flights.py /input/randomData100k.csv /output/
7. For adding the output to text file, execute this command - 
hadoop fs -cat output/* > output/true_pred100k.txt
8. Finally copied the output file to bucket and then downloaded directly from S3. For copying file to s3, ran command as -
aws s3 cp output/true_pred100k.txt s3://BUCKET_NAME/true_pred100k.txt