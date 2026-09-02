# BLAKE3
Pure Java&trade; implementation of <a href="https://github.com/BLAKE3-team/BLAKE3">BLAKE3</a> cryptographic hash function.

Main features:

- Extends <a href="https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/security/MessageDigest.html">MessageDigest</a> class and includes <a href="https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/security/Provider.html">Provider</a> method.
- SIMD and scalar compressor modes with automatic fallback.
- SIMD vectorization of blocks, chunks and parallel chunks.
- SIMD features scale with the instruction set of hardware.
- Configurable: enable or disable SIMD compressor.

DISCLAIMER: this is free and open source software. The developer takes no
responsibility for any personal, material or other damage the use of this
software might cause.

This software was developed and tested using <a href="https://docs.oracle.com/en/java/javase/25/">Java 25</a>.

Original API documentation is available <a href="https://www.eternitymud.com/doc/java/com/eternitymud/core/util/Blake3.html">here</a>.
