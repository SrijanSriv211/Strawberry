import regex, os

# save the text in a text file
ansi_escape = regex.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def print0(*text, println=True, overwrite=False, save_to_file=True, log_path="bin"):
	if println:
		print(*text)

	if not save_to_file:
		return

	# save cleaned text to the file
	if not os.path.isdir(log_path):
		os.mkdir(log_path)

	with open(os.path.join(log_path, "out.txt"), "w" if overwrite else "a", encoding="utf-8") as f:
		f.write(" ".join(tuple(ansi_escape.sub('', part) for part in text)) + "\n")

# cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
def print_banner():
	banner = """
          █████                                        █████                                            
         ░░███                                        ░░███                                             
  █████  ███████   ████████   ██████   █████ ███ █████ ░███████   ██████  ████████  ████████  █████ ████
 ███░░  ░░░███░   ░░███░░███ ░░░░░███ ░░███ ░███░░███  ░███░░███ ███░░███░░███░░███░░███░░███░░███ ░███ 
░░█████   ░███     ░███ ░░░   ███████  ░███ ░███ ░███  ░███ ░███░███████  ░███ ░░░  ░███ ░░░  ░███ ░███ 
 ░░░░███  ░███ ███ ░███      ███░░███  ░░███████████   ░███ ░███░███░░░   ░███      ░███      ░███ ░███ 
 ██████   ░░█████  █████    ░░████████  ░░████░████    ████████ ░░██████  █████     █████     ░░███████ 
░░░░░░     ░░░░░  ░░░░░      ░░░░░░░░    ░░░░ ░░░░    ░░░░░░░░   ░░░░░░  ░░░░░     ░░░░░       ░░░░░███ 
                                                                                               ███ ░███ 
                                                                                              ░░██████  
                                                                                               ░░░░░░   
            """
	print0(banner, save_to_file=False)

def calc_total_time(seconds):
	# separate the integer part (for hours, minutes, and seconds) from the fractional part (for milliseconds)
	sec_int, millis = divmod(seconds, 1)
	millis = int(millis * 1000) # convert the fractional part to milliseconds

	min, sec = divmod(int(sec_int), 60)
	hour, min = divmod(min, 60)
	hours, minutes, seconds = int(hour), int(min), int(sec)

	t = [
		f"{hours} hour" + ("s" if hours > 1 else "") if hours > 0 else None,
		f"{minutes} minute" + ("s" if minutes > 1 else "") if minutes > 0 else None,
		f"{seconds} second" + ("s" if seconds > 1 else "") if seconds > 0 else None,
		f"{millis} ms" if millis > 0 else None
	]
	t = list(filter(None, t))

	return ", ".join(t) if t else "0 seconds"
