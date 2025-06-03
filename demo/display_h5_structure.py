import h5py
import argparse
from humanize import naturalsize

class H5Printer:
    def __init__(self):
        self.entry_count = 0
        self.max_entries = 1000

    def print_h5_structure(self, file_path, min_size_kb=10, max_depth=3):
        """Display hierarchical structure of H5 file with size filtering"""
        try:
            with h5py.File(file_path, 'r') as h5_file:
                print(f"\n📁 File: {file_path}")
                print(f"🔍 Displaying structure (min_size={min_size_kb}KB, max_depth={max_depth})")
                self._print_h5_contents(h5_file, min_size_kb * 1024, max_depth, current_depth=0)
                print("\n📊 File Summary:")
                self._print_file_summary(h5_file)
        except Exception as e:
            print(f"❌ Error reading H5 file: {e}")

    def _print_h5_contents(self, item, min_size, max_depth, current_depth):
        """Recursive helper with size and depth filtering"""
        if current_depth > max_depth or self.entry_count >= self.max_entries:
            return

        indent = '  ' * current_depth

        if isinstance(item, (h5py.File, h5py.Group)):
            if self.entry_count < self.max_entries:
                print(f"{indent}📂 {item.name.split('/')[-1] or '/'}")
                self.entry_count += 1
            for key in item.keys():
                if self.entry_count < self.max_entries:
                    self._print_h5_contents(item[key], min_size, max_depth, current_depth + 1)
                else:
                    return

        elif isinstance(item, h5py.Dataset):
            if item.size * item.dtype.itemsize >= min_size and self.entry_count < self.max_entries:
                size_str = naturalsize(item.size * item.dtype.itemsize)
                print(f"{indent}📊 {item.name.split('/')[-1]} "
                      f"(shape={item.shape}, dtype={item.dtype}, size={size_str})")
                self.entry_count += 1

    def _print_file_summary(self, h5_file):
        """Print summary statistics about the file"""
        def _count_items(item):
            if isinstance(item, h5py.Dataset):
                return 1
            return sum(_count_items(item[key]) for key in item.keys())

        total_items = _count_items(h5_file)
        file_size = naturalsize(os.path.getsize(h5_file.filename))

        print(f"- Total size: {file_size}")
        print(f"- Total groups/datasets: {total_items}")
        print(f"- File format: {h5_file.driver}")
        print(f"- Access mode: {h5_file.mode}")
        if self.entry_count >= self.max_entries:
            print(f"\n⚠️ Display truncated after {self.max_entries} entries")

if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description='Display filtered HDF5 structure')
    parser.add_argument('--data-root',
                      default="/home/ubuntu/disk5/RePLAy",
                      help='Path to RePLAy Dataset')
    parser.add_argument('--min-size',
                      type=int,
                      default=10,
                      help='Minimum dataset size to show (KB)')
    parser.add_argument('--max-depth',
                      type=int,
                      default=3,
                      help='Maximum depth to traverse')
    parser.add_argument('--dataset',
                      type=str,
                      choices=["kitti", "kitti360", "nuscenes", "waymo", "ddad"],
                      default='ddad',
                      help='Dataset to analyze')

    args = parser.parse_args()

    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root directory not found: {args.data_root}")

    file_path = os.path.join(args.data_root, f"{args.dataset}.h5")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    printer = H5Printer()
    printer.print_h5_structure(file_path, args.min_size, args.max_depth)
