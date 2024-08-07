/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.imsi.queryEREngine.apache.calcite.sql.fun;

import java.math.BigDecimal;
import java.util.List;

import org.apache.calcite.linq4j.Ord;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.sql.SqlCall;
import org.imsi.queryEREngine.apache.calcite.sql.SqlCallBinding;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperandCountRange;
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperatorBinding;
import org.imsi.queryEREngine.apache.calcite.sql.SqlUtil;
import org.imsi.queryEREngine.apache.calcite.sql.SqlWriter;
import org.imsi.queryEREngine.apache.calcite.sql.type.FamilyOperandTypeChecker;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlOperandCountRanges;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeFamily;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeName;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeUtil;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlMonotonicity;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidator;

import com.google.common.collect.ImmutableList;

/**
 * Definition of the "SUBSTRING" builtin SQL function.
 */
public class SqlSubstringFunction extends SqlFunction {
	//~ Constructors -----------------------------------------------------------

	/**
	 * Creates the SqlSubstringFunction.
	 */
	SqlSubstringFunction() {
		super(
				"SUBSTRING",
				SqlKind.OTHER_FUNCTION,
				ReturnTypes.ARG0_NULLABLE_VARYING,
				null,
				null,
				SqlFunctionCategory.STRING);
	}

	//~ Methods ----------------------------------------------------------------

	@Override
	public String getSignatureTemplate(final int operandsCount) {
		switch (operandsCount) {
		case 2:
			return "{0}({1} FROM {2})";
		case 3:
			return "{0}({1} FROM {2} FOR {3})";
		default:
			throw new AssertionError();
		}
	}

	@Override
	public String getAllowedSignatures(String opName) {
		StringBuilder ret = new StringBuilder();
		for (Ord<SqlTypeName> typeName : Ord.zip(SqlTypeName.STRING_TYPES)) {
			if (typeName.i > 0) {
				ret.append(NL);
			}
			ret.append(
					SqlUtil.getAliasedSignature(this, opName,
							ImmutableList.of(typeName.e, SqlTypeName.INTEGER)));
			ret.append(NL);
			ret.append(
					SqlUtil.getAliasedSignature(this, opName,
							ImmutableList.of(typeName.e, SqlTypeName.INTEGER,
									SqlTypeName.INTEGER)));
		}
		return ret.toString();
	}

	@Override
	public boolean checkOperandTypes(
			SqlCallBinding callBinding,
			boolean throwOnFailure) {
		List<SqlNode> operands = callBinding.operands();
		int n = operands.size();
		assert (3 == n) || (2 == n);
		if (2 == n) {
			return OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.NUMERIC)
					.checkOperandTypes(callBinding, throwOnFailure);
		} else {
			final FamilyOperandTypeChecker checker1 = OperandTypes.STRING_STRING_STRING;
			final FamilyOperandTypeChecker checker2 = OperandTypes.family(
					SqlTypeFamily.STRING,
					SqlTypeFamily.NUMERIC,
					SqlTypeFamily.NUMERIC);
			// Put the STRING_NUMERIC_NUMERIC checker first because almost every other type
			// can be coerced to STRING.
			if (!OperandTypes.or(checker2, checker1)
					.checkOperandTypes(callBinding, throwOnFailure)) {
				return false;
			}
			// Reset the operands because they may be coerced during
			// implicit type coercion.
			operands = callBinding.getCall().getOperandList();
			final SqlValidator validator = callBinding.getValidator();
			final RelDataType t1 = validator.deriveType(callBinding.getScope(), operands.get(1));
			final RelDataType t2 = validator.deriveType(callBinding.getScope(), operands.get(2));
			if (SqlTypeUtil.inCharFamily(t1)) {
				if (!SqlTypeUtil.isCharTypeComparable(callBinding, operands,
						throwOnFailure)) {
					return false;
				}
			}
			if (!SqlTypeUtil.inSameFamily(t1, t2)) {
				if (throwOnFailure) {
					throw callBinding.newValidationSignatureError();
				}
				return false;
			}
		}
		return true;
	}

	@Override
	public SqlOperandCountRange getOperandCountRange() {
		return SqlOperandCountRanges.between(2, 3);
	}

	@Override
	public void unparse(
			SqlWriter writer,
			SqlCall call,
			int leftPrec,
			int rightPrec) {
		final SqlWriter.Frame frame = writer.startFunCall(getName());
		call.operand(0).unparse(writer, leftPrec, rightPrec);
		writer.sep("FROM");
		call.operand(1).unparse(writer, leftPrec, rightPrec);

		if (3 == call.operandCount()) {
			writer.sep("FOR");
			call.operand(2).unparse(writer, leftPrec, rightPrec);
		}

		writer.endFunCall(frame);
	}

	@Override public SqlMonotonicity getMonotonicity(SqlOperatorBinding call) {
		// SUBSTRING(x FROM 0 FOR constant) has same monotonicity as x
		if (call.getOperandCount() == 3) {
			final SqlMonotonicity mono0 = call.getOperandMonotonicity(0);
			if ((mono0 != SqlMonotonicity.NOT_MONOTONIC)
					&& call.getOperandMonotonicity(1) == SqlMonotonicity.CONSTANT
					&& call.getOperandLiteralValue(1, BigDecimal.class)
					.equals(BigDecimal.ZERO)
					&& call.getOperandMonotonicity(2) == SqlMonotonicity.CONSTANT) {
				return mono0.unstrict();
			}
		}
		return super.getMonotonicity(call);
	}
}
