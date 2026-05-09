def usage_description(description):
    """Formatting for description"""
    s = f"\n\nDESCRIPTION" f"\n\t{description}"
    return s


def usage_header(header):
    """Formatting for header"""
    s = f"\n\n{header.upper()}"
    return s


def add_option(option, description):
    """Add formatted option in description"""
    s = f"\n\n{option}," f"\n\t{description}"
    return s


def usage(argv):
    """Print program usage"""
    options = (
        ("-c/--config FILE", "specify configuration file", True),
        ("-h/--help", "print this help message", False),
        ("-s/--save", "save live footage", False),
        ("-w/--web PORT", "Start web server on port", False),
        ("--ha-trigger", "Home Assistant: trigger while person detected", False),
    )

    str_options = " ".join(
        [f"{opt[0]}" if opt[2] is True else f"[{opt[0]}]" for opt in options]
    )
    program_usage = f"{argv[0]} {str_options}"
    program_description = usage_description("Detect people from RTSP stream.")
    program_options = "".join([add_option(opt[0], opt[1]) for opt in options])

    program_help = (
        program_usage,
        program_description,
        usage_header("options"),
        program_options,
    )
    print("".join(program_help))
