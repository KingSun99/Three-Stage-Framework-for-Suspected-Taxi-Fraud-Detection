import hashlib

def generator_digest(generator):
    digest = hashlib.sha256()
    for name, buffer in sorted(generator.named_buffers()):
        digest.update(name.encode())
        digest.update(buffer.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
