/* Generated by: ParserGeneratorCC: Do not edit this line. AbstractCharStream.java Version 1.1 */
/* ParserGeneratorCCOptions:SUPPORT_CLASS_VISIBILITY_PUBLIC=true */
package org.imsi.queryEREngine.imsi.calcite.sql.parser.impl;

/**
 * An implementation of interface CharStream, where the stream is assumed to
 * contain only ASCII characters (without unicode processing).
 */

public
abstract class AbstractCharStream
implements CharStream
{
	/** Default buffer size if nothing is specified */
	public static final int DEFAULT_BUF_SIZE = 4096;
	/** By how much should the buffer be increase? */
	protected static final int BUFFER_INCREASE = 2048;

	static final int hexval(final char c) throws java.io.IOException {
		switch(c)
		{
		case '0' :
			return 0;
		case '1' :
			return 1;
		case '2' :
			return 2;
		case '3' :
			return 3;
		case '4' :
			return 4;
		case '5' :
			return 5;
		case '6' :
			return 6;
		case '7' :
			return 7;
		case '8' :
			return 8;
		case '9' :
			return 9;
		case 'a' :
		case 'A' :
			return 10;
		case 'b' :
		case 'B' :
			return 11;
		case 'c' :
		case 'C' :
			return 12;
		case 'd' :
		case 'D' :
			return 13;
		case 'e' :
		case 'E' :
			return 14;
		case 'f' :
		case 'F' :
			return 15;
		}

		// Should never come here
		throw new java.io.IOException("Invalid hex char '" + c + "' provided!");
	}

	/** Position in buffer. */
	protected int bufpos = -1;
	protected int bufsize;
	protected int available;
	protected int tokenBegin;

	protected char[] buffer;
	protected int inBuf = 0;
	private int tabSize = 1;
	protected int maxNextCharInd = 0;

	protected int[] bufline;
	protected int[] bufcolumn;

	protected int column = 0;
	protected int line = 1;

	protected boolean prevCharIsCR = false;
	protected boolean prevCharIsLF = false;
	private boolean trackLineColumn = true;

	@Override
	public void setTabSize(final int i)
	{
		tabSize = i;
	}

	@Override
	public int getTabSize()
	{
		return tabSize;
	}

	protected void expandBuff(final boolean wrapAround)
	{
		final char[] newbuffer = new char[bufsize + BUFFER_INCREASE];
		final int newbufline[] = new int[bufsize + BUFFER_INCREASE];
		final int newbufcolumn[] = new int[bufsize + BUFFER_INCREASE];

		try
		{
			if (wrapAround)
			{
				System.arraycopy(buffer, tokenBegin, newbuffer, 0, bufsize - tokenBegin);
				System.arraycopy(buffer, 0, newbuffer, bufsize - tokenBegin, bufpos);
				buffer = newbuffer;

				System.arraycopy(bufline, tokenBegin, newbufline, 0, bufsize - tokenBegin);
				System.arraycopy(bufline, 0, newbufline, bufsize - tokenBegin, bufpos);
				bufline = newbufline;

				System.arraycopy(bufcolumn, tokenBegin, newbufcolumn, 0, bufsize - tokenBegin);
				System.arraycopy(bufcolumn, 0, newbufcolumn, bufsize - tokenBegin, bufpos);
				bufcolumn = newbufcolumn;

				bufpos += (bufsize - tokenBegin);
				// https://github.com/phax/ParserGeneratorCC/pull/23
				// maxNextCharInd = bufpos;
			}
			else
			{
				System.arraycopy(buffer, tokenBegin, newbuffer, 0, bufsize - tokenBegin);
				buffer = newbuffer;

				System.arraycopy(bufline, tokenBegin, newbufline, 0, bufsize - tokenBegin);
				bufline = newbufline;

				System.arraycopy(bufcolumn, tokenBegin, newbufcolumn, 0, bufsize - tokenBegin);
				bufcolumn = newbufcolumn;

				bufpos -= tokenBegin;
				// https://github.com/phax/ParserGeneratorCC/pull/23
				// maxNextCharInd = bufpos;
			}
		}
		catch (final Exception ex)
		{
			throw new IllegalStateException(ex);
		}

		bufsize += BUFFER_INCREASE;
		available = bufsize;
		tokenBegin = 0;
	}

	protected abstract int streamRead(char[] aBuf, int nOfs, int nLen) throws java.io.IOException;

	protected abstract void streamClose() throws java.io.IOException;

	protected void fillBuff() throws java.io.IOException
	{
		if (maxNextCharInd == available)
		{
			if (available == bufsize)
			{
				if (tokenBegin > 2048)
				{
					bufpos = 0;
					maxNextCharInd = 0;
					available = tokenBegin;
				}
				else
					if (tokenBegin < 0)
					{
						bufpos = 0;
						maxNextCharInd = 0;
					}
					else
						expandBuff(false);
			}
			else
				if (available > tokenBegin)
					available = bufsize;
				else
					if ((tokenBegin - available) < 2048)
						expandBuff(true);
					else
						available = tokenBegin;
		}

		try
		{
			final int i = streamRead(buffer, maxNextCharInd, available - maxNextCharInd);
			if (i == -1)
			{
				streamClose();
				throw new java.io.IOException();
			}
			maxNextCharInd += i;
		}
		catch (final java.io.IOException ex)
		{
			--bufpos;
			backup(0);
			if (tokenBegin == -1)
				tokenBegin = bufpos;
			throw ex;
		}
	}

	@Override
	public char beginToken() throws java.io.IOException
	{
		tokenBegin = -1;
		final char c = readChar();
		tokenBegin = bufpos;
		return c;
	}

	protected void updateLineColumn(char c)
	{
		column++;

		if (prevCharIsLF)
		{
			prevCharIsLF = false;
			column = 1;
			line++;
		}
		else
			if (prevCharIsCR)
			{
				prevCharIsCR = false;
				if (c == '\n')
					prevCharIsLF = true;
				else
				{
					column = 1;
					line++;
				}
			}

		switch (c)
		{
		case '\r' :
			prevCharIsCR = true;
			break;
		case '\n' :
			prevCharIsLF = true;
			break;
		case '\t' :
			column--;
			column += (tabSize - (column % tabSize));
			break;
		default :
			break;
		}

		bufline[bufpos] = line;
		bufcolumn[bufpos] = column;
	}

	/** Read a character. */
	@Override
	public char readChar() throws java.io.IOException
	{
		if (inBuf > 0)
		{
			--inBuf;

			++bufpos;
			if (bufpos == bufsize)
				bufpos = 0;

			return buffer[bufpos];
		}

		++bufpos;
		if (bufpos >= maxNextCharInd)
			fillBuff();

		char c = buffer[bufpos];

		if (trackLineColumn)
			updateLineColumn(c);
		return c;
	}

	@Override
	public int getBeginColumn() {
		return bufcolumn[tokenBegin];
	}

	@Override
	public int getBeginLine() {
		return bufline[tokenBegin];
	}

	@Override
	public int getEndColumn() {
		return bufcolumn[bufpos];
	}

	@Override
	public int getEndLine() {
		return bufline[bufpos];
	}

	@Override
	public void backup(final int amount) {
		inBuf += amount;
		bufpos -= amount;
		if (bufpos < 0)
			bufpos += bufsize;
	}

	/** Constructor. */
	public AbstractCharStream(final int startline,
			final int startcolumn,
			final int buffersize)
	{
		line = startline;
		column = startcolumn - 1;

		bufsize = buffersize;
		available = buffersize;
		buffer = new char[buffersize];
		bufline = new int[buffersize];
		bufcolumn = new int[buffersize];
	}

	/** Reinitialise. */
	public void reInit(final int startline,
			final int startcolumn,
			final int buffersize)
	{
		line = startline;
		column = startcolumn - 1;
		prevCharIsCR = false;
		prevCharIsLF = false;
		if (buffer == null || buffersize != buffer.length)
		{
			bufsize = buffersize;
			available = buffersize;
			buffer = new char[buffersize];
			bufline = new int[buffersize];
			bufcolumn = new int[buffersize];
		}
		maxNextCharInd = 0;
		inBuf = 0;
		tokenBegin = 0;
		bufpos = -1;
	}

	@Override
	public String getImage()
	{
		if (bufpos >= tokenBegin)
		{
			// from tokenBegin to bufpos
			return new String(buffer, tokenBegin, bufpos - tokenBegin + 1);
		}

		// from tokenBegin to bufpos including wrap around
		return new String(buffer, tokenBegin, bufsize - tokenBegin) +
				new String(buffer, 0, bufpos + 1);
	}

	@Override
	public char[] getSuffix(final int len)
	{
		char[] ret = new char[len];
		if ((bufpos + 1) >= len)
			System.arraycopy(buffer, bufpos - len + 1, ret, 0, len);
		else
		{
			System.arraycopy(buffer, bufsize - (len - bufpos - 1), ret, 0, len - bufpos - 1);
			System.arraycopy(buffer, 0, ret, len - bufpos - 1, bufpos + 1);
		}
		return ret;
	}

	@Override
	public void done()
	{
		buffer = null;
		bufline = null;
		bufcolumn = null;
	}

	/**
	 * Method to adjust line and column numbers for the start of a token.
	 */
	public void adjustBeginLineColumn(final int nNewLine, final int newCol)
	{
		int start = tokenBegin;
		int newLine = nNewLine;
		int len;

		if (bufpos >= tokenBegin)
		{
			len = bufpos - tokenBegin + inBuf + 1;
		}
		else
		{
			len = bufsize - tokenBegin + bufpos + 1 + inBuf;
		}

		int i = 0;
		int j = 0;
		int k = 0;
		int nextColDiff = 0;
		int columnDiff = 0;

		// TODO disassemble meaning and split up
		while (i < len && bufline[j = start % bufsize] == bufline[k = ++start % bufsize])
		{
			bufline[j] = newLine;
			nextColDiff = columnDiff + bufcolumn[k] - bufcolumn[j];
			bufcolumn[j] = newCol + columnDiff;
			columnDiff = nextColDiff;
			i++;
		}

		if (i < len)
		{
			bufline[j] = newLine++;
			bufcolumn[j] = newCol + columnDiff;

			while (i++ < len)
			{
				// TODO disassemble meaning and split up
				if (bufline[j = start % bufsize] != bufline[++start % bufsize])
					bufline[j] = newLine++;
				else
					bufline[j] = newLine;
			}
		}

		line = bufline[j];
		column = bufcolumn[j];
	}

	@Override
	public void setTrackLineColumn(final boolean tlc) {
		trackLineColumn = tlc;
	}

	@Override
	public boolean isTrackLineColumn() {
		return trackLineColumn;
	}
}
/* ParserGeneratorCC - OriginalChecksum=e11e28132a9dde7cf84fa92e679226e9 (do not edit this line) */