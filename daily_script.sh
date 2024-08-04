 #!/bin/bash

 cd /home/chivier/Projects/darxiv
 DATEVAL=$(date +%Y-%m-%d)
 echo $DATEVAL

 /home/chivier/opt/miniconda3/envs/darxiv/bin/python darxiv/DailyArxiv.py --paper_dir="./data/paper" --ocr_dir="./data/ocr" --metadata_dir="./data/metadata" --index_dir="./data/index"
 rsync -avzP --partial -r data/index/index_$DATEVAL victorique:Projects/autoreader-backend/index
 rsync -avzP --partial -r data/metadata/papers_$DATEVAL.json victorique:Projects/autoreader-backend/metadata
