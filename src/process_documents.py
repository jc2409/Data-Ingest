import os, json
import pathlib
from dotenv import load_dotenv
load_dotenv()

import unstructured_client
from unstructured_client.models import operations, shared
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class DocumentProcessing():
    def __init__(self, max_workers=5):
        self.client = unstructured_client.UnstructuredClient(
            api_key_auth=os.getenv("UNSTRUCTURED_API_KEY")
        )
        self.max_workers = max_workers
        self.files = [os.getcwd() + '/dataset/src/' + file for file in os.listdir(os.getcwd()+ '/dataset/src')]

    def _get_optimal_parameters(self, filename):
        """Get optimal processing parameters based on file type"""
        ext = pathlib.Path(filename).suffix.lower()

        # Base parameters for all files
        params = {
            "languages": ['eng'],
        }

        # PDF-specific: Use hi_res for better image extraction with base64
        if ext == '.pdf':
            params.update({
                "strategy": shared.Strategy.HI_RES,
                "hi_res_model_name": "yolox",  # Layout detection model
                "pdf_infer_table_structure": True,  # Accurate table detection
                "extract_image_block_types": ["Image", "Table"],  # Extract images & tables as base64
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 10,
            })

        # PPTX: Use hi_res for complex layouts and images
        elif ext == '.pptx':
            params.update({
                "strategy": shared.Strategy.HI_RES,
                "extract_image_block_types": ["Image", "Table"],
                "infer_table_structure": True,
            })

        # DOCX: Use hi_res for quality, captures tables and formatting
        elif ext == '.docx':
            params.update({
                "strategy": shared.Strategy.HI_RES,
                "infer_table_structure": True,
            })

        # HTML: Auto strategy preserves structure
        elif ext == '.html':
            params.update({
                "strategy": shared.Strategy.AUTO,
            })

        # Simple text formats: Fast strategy is sufficient
        elif ext in ['.txt', '.md', '.csv']:
            params.update({
                "strategy": shared.Strategy.FAST,
            })

        # Default: Auto strategy
        else:
            params.update({
                "strategy": shared.Strategy.AUTO,
            })

        return params

    def process_single_file(self, filename):
        """Process a single file and return the result"""
        try:
            # Get optimal parameters for this file type
            params = self._get_optimal_parameters(filename)

            # Use context manager to ensure file is properly closed
            with open(filename, "rb") as file_content:
                req = operations.PartitionRequest(
                    partition_parameters=shared.PartitionParameters(
                        files=shared.Files(
                            content=file_content.read(),
                            file_name=filename,
                        ),
                        **params  # Unpack optimal parameters
                    ),
                )

            res = self.client.general.partition(request=req)
            element_dicts = [element for element in res.elements]

            # Write the processed data to a local file
            json_elements = json.dumps(element_dicts, indent=2)
            output_filename = filename.replace(pathlib.Path(filename).suffix, ".json")
            output_filename = output_filename.replace("/src/", "/res/")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

            with open(output_filename, "w") as file:
                file.write(json_elements)

            return {
                "status": "success",
                "filename": os.path.basename(filename),
                "output": output_filename,
                "elements_count": len(element_dicts),
                "strategy": params.get("strategy", "AUTO").value if hasattr(params.get("strategy", "AUTO"), "value") else str(params.get("strategy", "AUTO"))
            }
        except Exception as e:
            return {
                "status": "failed",
                "filename": os.path.basename(filename),
                "error": str(e)
            }

    def process_documents(self):
        """Process all documents with parallel processing and progress bar"""
        results = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, filename): filename
                for filename in self.files
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(self.files), desc="Processing documents", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    result = future.result()
                    results.append(result)

                    # Update progress bar description with status
                    if result["status"] == "success":
                        pbar.set_postfix_str(f"✓ {result['filename']}")
                    else:
                        pbar.set_postfix_str(f"✗ {result['filename']}")

                    pbar.update(1)

        # Print summary
        self._print_summary(results)
        return results

    def _print_summary(self, results):
        """Print processing summary"""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]

        print("\n" + "="*60)
        print(f"Processing Summary:")
        print(f"  Total files: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")

        if successful:
            print(f"\n✓ Successfully processed:")
            for r in successful:
                strategy = r.get('strategy', 'UNKNOWN')
                print(f"  - {r['filename']:40} | {strategy:8} | {r['elements_count']:4} elements")

        if failed:
            print(f"\n✗ Failed:")
            for r in failed:
                print(f"  - {r['filename']}: {r['error']}")

        print("="*60)

if __name__ == "__main__":
    # Configure max_workers based on your API rate limits
    # Default is 5 concurrent workers
    processor = DocumentProcessing(max_workers=5)
    processor.process_documents()