BEGIN {
    FS = ",";  # Sets the field separator to comma
    print "Calculating average expression per condition..."
}

NR > 1 {  # Skips the header line
    sum[$3] += $2;
    count[$3]++;
}

END {
    for (condition in sum) {
        average = sum[condition] / count[condition];
        print condition " average expression: " average;
    }
}
