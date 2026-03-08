
"attributes.txt" specifies the Transient Attributes and their order for the "annotations.tsv file"

"annotations.tsv" contains the aggregated scores and confidence for each attribute/image pair in the following format:

$imageName	  $attribute1_score,$attribute1_confidence	... 	 $attribute40_score,$attribute40_confidence

Each line corresponds to one image and the annotations for each attribute are tab separated. Each annotation is an ordered pair with the score and confidence for that attribute/image pair.

The scores are in the range of 0 to 1, with <0.2 being considered a strong
negative and >0.8 being considered a strong positive.

The confidence are the number of confident MT votes for that attribute/image. We consider at least 5 confident votes to be a reliable annotation.

Less than 1 percent of the attribute/image pairs have less than 5 confident votes.
