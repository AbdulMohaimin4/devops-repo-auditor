import argparse
from urllib.parse import urlparse
import os


def validate(repo, output):
    parsed = urlparse(repo)

    if parsed.scheme != "https" or parsed.netloc != "github.com":
        return False

    if len(parsed.path.strip("/").split("/")) != 2:
        return False

    if not output.endswith(".json"):
        return False

    return True



parser = argparse.ArgumentParser(description="clones the repo, identifies DevOps best-practice violations, \
                                 and generates an evidence-based report with actionable recommendations")
parser.add_argument("--repo", 
                    help="URL of a public Github repository", 
                    type=str)
parser.add_argument("--output",
                    help="name of the output json file",
                    type=str)
args = parser.parse_args()


if __name__ == "__main__":
    if args.repo and args.output:
        if validate(args.repo, args.output):
            print(f"repo URL: {args.repo}\noutput file: {args.output}")
        else:
            print("Invalid GitHub URL or JSON filename")
    else:
        print("Github repo or output file name missing")

